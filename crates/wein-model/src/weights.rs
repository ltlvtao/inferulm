//! safetensors 手写解析 + 多 shard 路由。
//!
//! 格式：`[8B LE 头长][JSON 元数据][裸数据]`。元数据里每张量带 dtype、
//! shape、data_offsets（相对数据区）。多 shard 时按官方
//! `model.safetensors.index.json` 的 weight_map 路由；单文件模型直接开
//! `model.safetensors`。mmap 零整拷——权重只读，上载设备时按张量切片。

use crate::{ModelError, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F32,
}

impl Dtype {
    fn parse(s: &str) -> std::result::Result<Dtype, String> {
        match s {
            "BF16" => Ok(Dtype::BF16),
            "F32" => Ok(Dtype::F32),
            other => Err(format!("不支持的 dtype：{other}（I1 只需 BF16/F32）")),
        }
    }
    pub fn elem_size(&self) -> usize {
        match self {
            Dtype::BF16 => 2,
            Dtype::F32 => 4,
        }
    }
}

/// 借用视图：名字解析后的张量（字节属于 mmap，WeightStore 存活期间有效）
pub struct TensorRef<'a> {
    pub dtype: Dtype,
    pub shape: Vec<u32>,
    pub bytes: &'a [u8],
}

impl TensorRef<'_> {
    /// bf16 数据转 u16 位形切片（对齐校验：mmap 页对齐 + 写出器按元素对齐）
    pub fn bf16(&self) -> Result<&[u16]> {
        if self.dtype != Dtype::BF16 {
            return Err(ModelError::Dtype {
                name: String::new(),
                want: Dtype::BF16,
                got: self.dtype,
            });
        }
        if self.bytes.as_ptr() as usize % 2 != 0 {
            return Err(ModelError::Unaligned("bf16".into()));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.bytes.as_ptr() as *const u16, self.bytes.len() / 2) })
    }

    /// f32 数据切片
    pub fn f32(&self) -> Result<&[f32]> {
        if self.dtype != Dtype::F32 {
            return Err(ModelError::Dtype {
                name: String::new(),
                want: Dtype::F32,
                got: self.dtype,
            });
        }
        if self.bytes.as_ptr() as usize % 4 != 0 {
            return Err(ModelError::Unaligned("f32".into()));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.bytes.as_ptr() as *const f32, self.bytes.len() / 4) })
    }
}

#[derive(Debug)]
struct Entry {
    dtype: Dtype,
    shape: Vec<u32>,
    shard: usize,
    /// 相对数据区起点
    start: usize,
    len: usize,
}

#[derive(Deserialize)]
struct RawHeader {
    #[serde(flatten)]
    tensors: HashMap<String, RawTensor>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u32>,
    data_offsets: (usize, usize),
}

#[derive(Deserialize)]
struct RawIndex {
    weight_map: HashMap<String, String>,
}

pub struct WeightStore {
    maps: Vec<Mmap>,
    /// 每 shard 数据区起点（8 + 头长）
    data_bases: Vec<usize>,
    tensors: HashMap<String, Entry>,
    dir: PathBuf,
}

impl WeightStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shard_files: Vec<PathBuf> = Vec::new();
        let index = dir.join("model.safetensors.index.json");
        if index.is_file() {
            let raw: RawIndex = serde_json::from_str(&std::fs::read_to_string(&index)?)
                .map_err(|e| ModelError::Json(format!("index: {e}")))?;
            let mut seen = std::collections::BTreeSet::new();
            for file in raw.weight_map.values() {
                if seen.insert(file.clone()) {
                    shard_files.push(dir.join(file));
                }
            }
            shard_files.sort();
        } else {
            let single = dir.join("model.safetensors");
            if !single.is_file() {
                return Err(ModelError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("{} 既无 index 也无 model.safetensors", dir.display()),
                )));
            }
            shard_files.push(single);
        }

        let mut maps = Vec::new();
        let mut data_bases = Vec::new();
        let mut tensors = HashMap::new();
        for (si, path) in shard_files.iter().enumerate() {
            let file = std::fs::File::open(path)?;
            let map = unsafe { Mmap::map(&file)? };
            let header_len = u64::from_le_bytes(
                map.get(0..8)
                    .ok_or_else(|| ModelError::Json(format!("{} 文件太短", path.display())))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            let header_json = map
                .get(8..8 + header_len)
                .ok_or_else(|| ModelError::Json(format!("{} 头越界", path.display())))?;
            let header: RawHeader = serde_json::from_slice(header_json)
                .map_err(|e| ModelError::Json(format!("{} 头解析：{e}", path.display())))?;
            let data_base = 8 + header_len;
            let data_len = map.len() - data_base;
            for (name, t) in header.tensors {
                let dtype = Dtype::parse(&t.dtype).map_err(ModelError::Json)?;
                let (s, e) = t.data_offsets;
                let len = e - s;
                if e > data_len || len != t.shape.iter().product::<u32>() as usize * dtype.elem_size() {
                    return Err(ModelError::Json(format!("{name}：偏移/长度与形状不符")));
                }
                tensors.insert(
                    name,
                    Entry { dtype, shape: t.shape, shard: si, start: s, len },
                );
            }
            data_bases.push(data_base);
            maps.push(map);
        }
        Ok(WeightStore { maps, data_bases, tensors, dir: dir.to_path_buf() })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        v.sort();
        v
    }

    pub fn tensor(&self, name: &str) -> Result<TensorRef<'_>> {
        let e = self.tensors.get(name).ok_or_else(|| ModelError::Missing(name.to_string()))?;
        let base = self.data_bases[e.shard] + e.start;
        let bytes = &self.maps[e.shard][base..base + e.len];
        Ok(TensorRef { dtype: e.dtype, shape: e.shape.clone(), bytes })
    }
}
