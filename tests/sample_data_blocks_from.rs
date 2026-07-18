#[test_log::test]
fn sample_data_blocks_from_seeks_and_aborts() -> lsm_tree::Result<()> {
    use lsm_tree::{AbstractTree, SampleVerdict, SequenceNumberCounter};
    use std::cell::RefCell;

    let folder = tempfile::tempdir()?;
    let tree = lsm_tree::Config::new(
        folder.path(),
        SequenceNumberCounter::default(),
        SequenceNumberCounter::default(),
    )
    .data_block_size_policy(lsm_tree::config::BlockSizePolicy::all(1_024))
    .open()?;

    // three "collections" of 100 keys each, interleaved collection prefixes
    for i in 0..100u32 {
        for collection in ["a", "b", "c"] {
            let key = format!("{collection}|{i:04}");
            tree.insert(key, format!("value-{collection}-{i}"), 0);
        }
    }
    tree.flush_active_memtable(0)?;
    tree.major_compact(u64::MAX, 0)?;

    let total_blocks = tree.sample_data_blocks(usize::MAX, |_, _| true)?.len();
    assert!(
        total_blocks > 0,
        "expected some data blocks after compaction"
    );

    // no start key: same full walk as sample_data_blocks
    let all = tree
        .sample_data_blocks_from(None, usize::MAX, |_, _| SampleVerdict::Include)?
        .len();
    assert_eq!(all, total_blocks);

    // start key inside "b|": the predicate must never see keys below it
    let seen = RefCell::new(Vec::new());
    let from_b =
        tree.sample_data_blocks_from(Some(b"b|".as_slice()), usize::MAX, |first, last| {
            seen.borrow_mut().push((first.to_vec(), last.to_vec()));
            SampleVerdict::Include
        })?;
    assert!(!from_b.is_empty());
    assert!(from_b.len() < total_blocks);
    for (first, last) in seen.borrow().iter() {
        assert!(
            last.as_slice() >= b"b|".as_slice(),
            "block below start key was read"
        );
        assert!(first.as_slice() >= b"a".as_slice(), "sanity");
    }

    // abort: stop at the first block reaching "b|", collecting only "a" blocks
    let aborted_at = RefCell::new(None);
    let samples = tree.sample_data_blocks_from(None, usize::MAX, |first, _last| {
        if first >= b"b|".as_slice() {
            *aborted_at.borrow_mut() = Some(first.to_vec());
            SampleVerdict::Abort
        } else {
            SampleVerdict::Include
        }
    })?;
    let aborted_at = aborted_at.into_inner().expect("abort never fired");
    assert!(aborted_at.as_slice() >= b"b|".as_slice());
    assert!(
        samples.len() < total_blocks,
        "abort must stop the scan early"
    );

    Ok(())
}
