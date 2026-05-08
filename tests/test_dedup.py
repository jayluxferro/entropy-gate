"""Tests for context block deduplication."""

from entropy_gate.dedup import deduplicate_blocks, _hash_block, _normalize_block, _split_into_blocks


def test_normalize_block_whitespace():
    assert _normalize_block("hello   world") == "hello world"
    assert _normalize_block("  hello\n\nworld  ") == "hello world"


def test_hash_block_deterministic():
    assert _hash_block("hello world") == _hash_block("hello world")
    assert _hash_block("hello world") == _hash_block("hello   world")


def test_hash_block_different():
    assert _hash_block("hello world") != _hash_block("goodbye world")


def test_split_into_blocks_paragraphs():
    text = "Block A line 1.\n\nBlock B line 1.\n\nBlock C with enough characters to meet the minimum block size requirement."
    blocks = _split_into_blocks(text, min_block_chars=30)
    assert len(blocks) >= 2


def test_split_into_blocks_short_text():
    text = "short text"
    blocks = _split_into_blocks(text, min_block_chars=80)
    assert len(blocks) >= 1


def test_deduplicate_blocks_no_duplicates():
    text = "First paragraph with enough content to be a valid block.\n\nSecond paragraph also with sufficient content here."
    result = deduplicate_blocks(text, min_block_chars=20)
    assert result.blocks_removed == 0
    assert result.tokens_saved == 0
    assert len(result.deduplicated_text) > 0


def test_deduplicate_blocks_with_duplicates():
    block = "This is a repeated block with enough characters to meet the minimum block size requirement for deduplication testing."
    text = f"{block}\n\n{block}\n\n{block}"
    result = deduplicate_blocks(text, min_block_chars=20)
    assert result.blocks_removed == 2
    assert result.tokens_saved > 0


def test_deduplicate_blocks_preserves_first_occurrence():
    unique = "Unique first block with enough content length to pass minimum requirements."
    repeated = "Repeated block content that appears multiple times with enough length."
    text = f"{unique}\n\n{repeated}\n\n{repeated}"
    result = deduplicate_blocks(text, min_block_chars=20)
    assert "Unique" in result.deduplicated_text
    assert result.blocks_removed == 1


def test_deduplicate_blocks_short_blocks_not_deduped():
    text = "hi\n\nhi\n\nhi"
    result = deduplicate_blocks(text, min_block_chars=80)
    assert result.blocks_removed == 0


def test_deduplicate_blocks_single_block():
    text = "just one block"
    result = deduplicate_blocks(text)
    assert result.blocks_removed == 0
