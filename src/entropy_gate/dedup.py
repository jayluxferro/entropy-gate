"""Context block deduplication pre-pass.

Collapses identical or near-identical blocks repeated across messages
before the cooling stage, addressing the token "amnesia tax."
"""

import hashlib
import re

from entropy_gate.models import DedupBlock, DedupResult


def _normalize_block(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, strip trailing blanks."""
    return re.sub(r"\s+", " ", text).strip()


def _hash_block(text: str) -> str:
    """Content-addressable hash for a text block."""
    normalized = _normalize_block(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _split_into_blocks(text: str, min_block_chars: int = 80) -> list[str]:
    """Split text into dedup-able blocks at paragraph boundaries.

    Blocks shorter than min_block_chars are not deduplicated (too noisy).
    """
    # Split on double newlines (paragraph boundaries)
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks: list[str] = []
    small_buf: list[str] = []

    for block in raw_blocks:
        if len(block) < min_block_chars:
            small_buf.append(block)
            # Merge accumulated small blocks if they grow large enough
            merged = "\n\n".join(small_buf)
            if len(merged) >= min_block_chars:
                blocks.append(merged)
                small_buf = []
        else:
            if small_buf:
                blocks.append("\n\n".join(small_buf))
                small_buf = []
            blocks.append(block)

    if small_buf:
        blocks.append("\n\n".join(small_buf))

    return blocks


def deduplicate_blocks(text: str, min_block_chars: int = 80) -> DedupResult:
    """Find and collapse repeated text blocks.

    The first occurrence of each block is kept; subsequent occurrences
    are collapsed to a short reference marker.

    Args:
        text: The concatenated text of all messages.
        min_block_chars: Minimum characters for a block to be dedup-eligible.

    Returns:
        DedupResult with deduplicated text and statistics.
    """
    blocks = _split_into_blocks(text, min_block_chars)
    if len(blocks) < 2:
        return DedupResult(
            deduplicated_text=text,
            blocks_removed=0,
            tokens_saved=0,
            blocks=[],
        )

    seen: dict[str, DedupBlock] = {}
    deduped_blocks: list[str] = []
    tokens_saved = 0

    for i, block_text in enumerate(blocks):
        block_hash = _hash_block(block_text)
        normalized = _normalize_block(block_text)

        if block_hash in seen:
            seen[block_hash].occurrences += 1
            seen[block_hash].indices.append(i)
            # Replace with a compact reference marker
            ref = f"[↳ see block {seen[block_hash].indices[0]}]"
            deduped_blocks.append(ref)
            tokens_saved += len(normalized.split()) - len(ref.split())
        else:
            seen[block_hash] = DedupBlock(
                hash=block_hash,
                text=normalized,
                occurrences=1,
                indices=[i],
            )
            deduped_blocks.append(normalized)

    deduplicated_text = "\n\n".join(deduped_blocks)

    return DedupResult(
        deduplicated_text=deduplicated_text,
        blocks_removed=sum(b.occurrences - 1 for b in seen.values()),
        tokens_saved=tokens_saved,
        blocks=list(seen.values()),
    )
