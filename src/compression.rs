// Copyright (c) 2024-present, fjall-rs
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

use crate::coding::{Decode, Encode};
use byteorder::{ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

#[cfg(feature = "zstd")]
use std::sync::Arc;

/// Compression algorithm to use
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompressionType {
    /// No compression
    ///
    /// Not recommended.
    None,

    /// LZ4 compression
    ///
    /// Recommended for use cases with a focus
    /// on speed over compression ratio.
    #[cfg(feature = "lz4")]
    Lz4,

    /// Zstandard compression
    ///
    /// Recommended for use cases with a focus on compression ratio.
    /// Level 1 is fastest, level 22 is best compression.
    /// Level 3 is the zstd default and a good starting point.
    #[cfg(feature = "zstd")]
    Zstd {
        /// Compression level (1–22; 3 is the zstd default).
        level: i32,
    },

    /// Zstandard compression with a pre-trained dictionary.
    ///
    /// The dictionary is embedded in the segment metadata, so each segment
    /// is self-contained and can be decompressed independently.
    ///
    /// Use [`train_zstd_dict`] to generate a dictionary from representative data.
    ///
    /// [`train_zstd_dict`]: crate::train_zstd_dict
    #[cfg(feature = "zstd")]
    ZstdDict {
        /// Compression level (1–22; 3 is the zstd default).
        level: i32,
        /// The pre-trained Zstandard dictionary bytes.
        dict: Arc<[u8]>,
    },
}

// ZstdDict contains Arc<[u8]> which is not Copy, so we implement Copy only
// for the variants that support it. The enum as a whole cannot be Copy.

impl std::fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::None => "none".to_owned(),

                #[cfg(feature = "lz4")]
                Self::Lz4 => "lz4".to_owned(),

                #[cfg(feature = "zstd")]
                Self::Zstd { level } => format!("zstd(level={level})"),

                #[cfg(feature = "zstd")]
                Self::ZstdDict { level, .. } => format!("zstd+dict(level={level})"),
            }
        )
    }
}

impl Encode for CompressionType {
    fn encode_into<W: Write>(&self, writer: &mut W) -> Result<(), crate::Error> {
        #[cfg(feature = "zstd")]
        use byteorder::LittleEndian;

        match self {
            Self::None => {
                writer.write_u8(0)?;
            }

            #[cfg(feature = "lz4")]
            Self::Lz4 => {
                writer.write_u8(1)?;
            }

            #[cfg(feature = "zstd")]
            Self::Zstd { level } => {
                writer.write_u8(2)?;
                writer.write_i32::<LittleEndian>(*level)?;
            }

            #[cfg(feature = "zstd")]
            Self::ZstdDict { level, dict } => {
                writer.write_u8(3)?;
                writer.write_i32::<LittleEndian>(*level)?;

                #[expect(clippy::cast_possible_truncation, reason = "dict is bounded")]
                writer.write_u32::<LittleEndian>(dict.len() as u32)?;

                writer.write_all(dict)?;
            }
        }

        Ok(())
    }
}

impl Decode for CompressionType {
    fn decode_from<R: Read>(reader: &mut R) -> Result<Self, crate::Error> {
        #[cfg(feature = "zstd")]
        use byteorder::LittleEndian;

        let tag = reader.read_u8()?;

        match tag {
            0 => Ok(Self::None),

            #[cfg(feature = "lz4")]
            1 => Ok(Self::Lz4),

            #[cfg(feature = "zstd")]
            2 => {
                let level = reader.read_i32::<LittleEndian>()?;
                Ok(Self::Zstd { level })
            }

            #[cfg(feature = "zstd")]
            3 => {
                let level = reader.read_i32::<LittleEndian>()?;
                let dict_len = reader.read_u32::<LittleEndian>()?;

                let mut dict = vec![0u8; dict_len as usize];
                reader.read_exact(&mut dict)?;

                Ok(Self::ZstdDict {
                    level,
                    dict: dict.into(),
                })
            }

            tag => Err(crate::Error::InvalidTag(("CompressionType", tag))),
        }
    }
}

/// Per-thread cache for pre-digested zstd dictionary contexts.
///
/// Building an [`EncoderDictionary`] or [`DecoderDictionary`] from raw bytes is
/// expensive (it calls `ZSTD_createCDict` / `ZSTD_createDDict` internally). This
/// module maintains one cached entry per thread per direction (encode / decode),
/// keyed on the Arc data pointer + length + compression level. Because we use
/// pointer identity the hit rate is 100 % as long as callers hold on to the same
/// `Arc<[u8]>` — which is the normal case: a tree opens once and keeps the same
/// `CompressionType::ZstdDict` for its lifetime.
///
/// Thread-local storage is used rather than a `Mutex`-protected global so that
/// compaction threads never contend with reader threads.
#[cfg(feature = "zstd")]
pub mod dict_cache {
    use std::{cell::RefCell, sync::Arc};
    use zstd::{
        bulk::{Compressor, Decompressor},
        dict::{DecoderDictionary, EncoderDictionary},
    };

    struct CachedEncoder {
        dict: EncoderDictionary<'static>,
        level: i32,
        // Pointer + length together uniquely identify the Arc allocation.
        // Arc guarantees the address is stable for the lifetime of any clone.
        ptr: *const u8,
        len: usize,
    }

    struct CachedDecoder {
        dict: DecoderDictionary<'static>,
        ptr: *const u8,
        len: usize,
    }

    // Safety: these structs are only ever accessed from their owning thread via
    // thread_local!. The raw pointer is never dereferenced after construction —
    // it is only compared for equality (cache-key identity check).
    unsafe impl Send for CachedEncoder {}
    unsafe impl Send for CachedDecoder {}

    thread_local! {
        static ENCODER: RefCell<Option<CachedEncoder>> = const { RefCell::new(None) };
        static DECODER: RefCell<Option<CachedDecoder>> = const { RefCell::new(None) };
    }

    /// Compress `data` using the given zstd dictionary, reusing the per-thread
    /// [`EncoderDictionary`] when the dict `Arc` matches the cached one.
    pub fn compress(data: &[u8], level: i32, dict: &Arc<[u8]>) -> std::io::Result<Vec<u8>> {
        ENCODER.with(|cell| {
            let mut slot = cell.borrow_mut();

            let hit = slot
                .as_ref()
                .is_some_and(|c| c.ptr == dict.as_ptr() && c.len == dict.len() && c.level == level);

            if !hit {
                *slot = Some(CachedEncoder {
                    dict: EncoderDictionary::copy(dict, level),
                    level,
                    ptr: dict.as_ptr(),
                    len: dict.len(),
                });
            }

            let mut compressor =
                Compressor::with_prepared_dictionary(&slot.as_ref().unwrap().dict)?;
            compressor.compress(data)
        })
    }

    /// Decompress `data` using the given zstd dictionary, reusing the per-thread
    /// [`DecoderDictionary`] when the dict `Arc` matches the cached one.
    pub fn decompress(data: &[u8], capacity: usize, dict: &Arc<[u8]>) -> std::io::Result<Vec<u8>> {
        DECODER.with(|cell| {
            let mut slot = cell.borrow_mut();

            let hit = slot
                .as_ref()
                .is_some_and(|c| c.ptr == dict.as_ptr() && c.len == dict.len());

            if !hit {
                *slot = Some(CachedDecoder {
                    dict: DecoderDictionary::copy(dict),
                    ptr: dict.as_ptr(),
                    len: dict.len(),
                });
            }

            let mut decompressor =
                Decompressor::with_prepared_dictionary(&slot.as_ref().unwrap().dict)?;
            decompressor.decompress(data, capacity)
        })
    }
}

/// Trains a Zstandard dictionary from a set of representative data samples.
///
/// The dictionary can be used with [`CompressionType::ZstdDict`] to improve
/// compression ratio for data with common structure.
///
/// A `dict_size` of 112 KiB (114_688 bytes) is a good starting point.
#[cfg(feature = "zstd")]
pub fn train_zstd_dict(samples: &[impl AsRef<[u8]>], dict_size: usize) -> crate::Result<Vec<u8>> {
    let sample_sizes: Vec<usize> = samples.iter().map(|s| s.as_ref().len()).collect();
    let flattened: Vec<u8> = samples
        .iter()
        .flat_map(|s| s.as_ref().iter().copied())
        .collect();
    zstd::dict::from_continuous(&flattened, &sample_sizes, dict_size).map_err(crate::Error::Io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn compression_serialize_none() {
        let serialized = CompressionType::None.encode_into_vec();
        assert_eq!(1, serialized.len());
    }

    #[cfg(feature = "lz4")]
    mod lz4 {
        use super::*;
        use test_log::test;

        #[test]
        fn compression_serialize_lz4() {
            let serialized = CompressionType::Lz4.encode_into_vec();
            assert_eq!(1, serialized.len());
        }
    }

    #[cfg(feature = "zstd")]
    mod zstd_tests {
        use super::*;
        use test_log::test;

        #[test]
        fn compression_serialize_zstd() {
            let c = CompressionType::Zstd { level: 3 };
            let serialized = c.encode_into_vec();
            // tag (1) + level i32 (4)
            assert_eq!(5, serialized.len());

            let mut reader = &serialized[..];
            let decoded = CompressionType::decode_from(&mut reader).unwrap();
            assert_eq!(decoded, CompressionType::Zstd { level: 3 });
        }

        #[test]
        fn compression_serialize_zstd_dict() {
            let dict: Arc<[u8]> = vec![1u8, 2, 3, 4].into();
            let c = CompressionType::ZstdDict {
                level: 3,
                dict: dict.clone(),
            };
            let serialized = c.encode_into_vec();
            // tag (1) + level i32 (4) + dict_len u32 (4) + dict bytes (4)
            assert_eq!(13, serialized.len());

            let mut reader = &serialized[..];
            let decoded = CompressionType::decode_from(&mut reader).unwrap();
            assert_eq!(decoded, CompressionType::ZstdDict { level: 3, dict });
        }
    }
}
