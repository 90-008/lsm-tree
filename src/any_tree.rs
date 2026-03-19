// Copyright (c) 2024-present, fjall-rs
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

use crate::{BlobTree, Tree};
use enum_dispatch::enum_dispatch;

/// May be a standard [`Tree`] or a [`BlobTree`]
#[derive(Clone)]
#[enum_dispatch(AbstractTree)]
pub enum AnyTree {
    /// Standard LSM-tree, see [`Tree`]
    Standard(Tree),

    /// Key-value separated LSM-tree, see [`BlobTree`]
    Blob(BlobTree),
}

impl AnyTree {
    /// Collects up to `limit` raw data block payloads from on-disk tables.
    ///
    /// See [`Tree::sample_data_blocks`] for details.
    ///
    /// # Errors
    ///
    /// Will return `Err` if an IO error occurs.
    pub fn sample_data_blocks<F: Fn(&[u8], &[u8]) -> bool>(
        &self,
        limit: usize,
        predicate: F,
    ) -> crate::Result<Vec<crate::Slice>> {
        match self {
            Self::Standard(tree) => tree.sample_data_blocks(limit, predicate),
            Self::Blob(blob_tree) => blob_tree.index.sample_data_blocks(limit, predicate),
        }
    }
}
