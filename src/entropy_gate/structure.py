"""Structural decomposition of chat-completion / messages bodies.

This module knows the shape of OpenAI (``/v1/chat/completions``) and
Anthropic (``/v1/messages``) request bodies and produces a normalized,
turn-indexed view that downstream compression can operate on safely.

Design invariants
-----------------
* The **last user message** is never altered — it is the live query.
* ``tool_use`` / ``tool_result`` blocks (Anthropic) and ``tool_calls`` /
  ``role=tool`` messages (OpenAI) are preserved **verbatim**. Compressing
  these breaks tool-call causality and corrupts agent state.
* ``system`` messages are preserved verbatim (defines agent capabilities).
* Only ``text`` blocks inside non-last user / assistant turns are eligible
  for compression. Everything else is "frozen".

This module is intentionally pure (no I/O, no model calls).  It returns a
plan that the proxy executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


CompressibleRole = Literal["user", "assistant"]


@dataclass
class CompressibleSpan:
    """A piece of text that may be compressed.

    ``message_index`` and ``block_index`` together identify where to write
    the compressed text back into the original body. ``block_index`` is
    ``None`` when the message has plain string ``content`` (OpenAI style
    or Anthropic with a top-level string).
    """

    message_index: int
    block_index: int | None
    role: CompressibleRole
    text: str
    turn_index: int  # 0 = oldest turn, increases toward recent
    is_last_user: bool = False  # never compressed; included for completeness


@dataclass
class MessagePlan:
    """A normalized view of the request used by the compression pass."""

    api: Literal["openai", "anthropic"]
    spans: list[CompressibleSpan] = field(default_factory=list)
    total_turns: int = 0

    # Spans that must not be touched (tool_use/tool_result/system/images/last user).
    # Kept implicitly via "not in spans"; this list is for debugging/visibility.
    frozen_message_indices: list[int] = field(default_factory=list)


def detect_api(body: dict[str, Any]) -> Literal["openai", "anthropic"]:
    """Detect API shape.

    Anthropic ``/v1/messages`` carries ``max_tokens`` as required and uses
    a top-level ``system`` *string* field (not a message).  OpenAI uses a
    ``system`` *message* in ``messages``.  When both are ambiguous we
    inspect message content shapes.
    """
    if isinstance(body.get("system"), str):
        return "anthropic"
    # Anthropic content blocks have explicit ``type`` keys; OpenAI multimodal
    # parts also do, but Anthropic mandates them for tool flows.
    for msg in body.get("messages", []) or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "tool_use",
                    "tool_result",
                ):
                    return "anthropic"
    # Anthropic disallows ``role=system`` in messages — its presence implies OpenAI.
    for msg in body.get("messages", []) or []:
        if msg.get("role") == "system":
            return "openai"
    # Default: OpenAI (the broader / older shape).
    return "openai"


def _is_compressible_text_block(block: Any) -> bool:
    """An Anthropic content block that is plain text and safe to compress."""
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _message_has_tool_payload(msg: dict[str, Any], api: str) -> bool:
    """True if this message carries tool-call state that must be preserved verbatim."""
    if api == "openai":
        if msg.get("role") in ("tool", "function"):
            return True
        if msg.get("tool_calls") or msg.get("function_call"):
            return True
        return False
    # anthropic
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "tool_use",
                "tool_result",
            ):
                return True
    return False


def body_has_signed_blocks(body: dict[str, Any]) -> bool:
    """True if the body contains any signed Anthropic content blocks.

    ``thinking`` and ``redacted_thinking`` blocks carry signatures that
    Anthropic validates against the exact JSON encoding it served.  Any
    JSON parse + re-serialize through the chain breaks them — even if
    the blocks themselves are passed through unchanged — so callers
    must forward raw bytes for these requests.
    """
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "thinking",
                    "redacted_thinking",
                ):
                    return True
    return False


def plan_compression(
    body: dict[str, Any],
    *,
    min_chars: int = 80,
) -> MessagePlan:
    """Walk the request body and produce a compression plan.

    ``min_chars`` — text blocks shorter than this are skipped (not worth
    the entropy-quench overhead and risk).
    """
    api = detect_api(body)
    messages = list(body.get("messages") or [])
    plan = MessagePlan(api=api, total_turns=len(messages))

    # Find the last user message — it is the live query, never compressed.
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            plan.frozen_message_indices.append(i)
            continue
        if _message_has_tool_payload(msg, api):
            plan.frozen_message_indices.append(i)
            continue
        if i == last_user_idx:
            # The live query — frozen.
            plan.frozen_message_indices.append(i)
            continue

        content = msg.get("content")
        if isinstance(content, str):
            if len(content) >= min_chars:
                plan.spans.append(
                    CompressibleSpan(
                        message_index=i,
                        block_index=None,
                        role=role,  # type: ignore[arg-type]
                        text=content,
                        turn_index=i,
                    )
                )
            else:
                plan.frozen_message_indices.append(i)
        elif isinstance(content, list):
            any_compressible = False
            for b_idx, block in enumerate(content):
                if not _is_compressible_text_block(block):
                    continue
                text = block.get("text", "")
                if len(text) < min_chars:
                    continue
                plan.spans.append(
                    CompressibleSpan(
                        message_index=i,
                        block_index=b_idx,
                        role=role,  # type: ignore[arg-type]
                        text=text,
                        turn_index=i,
                    )
                )
                any_compressible = True
            if not any_compressible:
                plan.frozen_message_indices.append(i)
        else:
            plan.frozen_message_indices.append(i)

    return plan


def apply_compression(
    body: dict[str, Any],
    plan: MessagePlan,
    replacements: dict[tuple[int, int | None], str],
) -> dict[str, Any]:
    """Return a new body with compressed text written back into the right slots.

    ``replacements`` maps ``(message_index, block_index)`` -> compressed text.
    Spans not present in the map are left untouched.  Original body is not
    mutated (defensive copy).
    """
    new_body = dict(body)
    src_messages = list(body.get("messages") or [])
    new_messages: list[dict[str, Any]] = []

    for i, msg in enumerate(src_messages):
        # Find replacements that target this message.
        msg_repls = {
            block_idx: text
            for (m_idx, block_idx), text in replacements.items()
            if m_idx == i
        }
        if not msg_repls:
            new_messages.append(msg)
            continue

        new_msg = dict(msg)
        content = msg.get("content")

        if None in msg_repls and isinstance(content, str):
            new_msg["content"] = msg_repls[None]
        elif isinstance(content, list):
            new_content = []
            for b_idx, block in enumerate(content):
                if b_idx in msg_repls and isinstance(block, dict):
                    new_block = dict(block)
                    new_block["text"] = msg_repls[b_idx]
                    new_content.append(new_block)
                else:
                    new_content.append(block)
            new_msg["content"] = new_content
        # else: replacement keyed to None but content isn't a string — skip.

        new_messages.append(new_msg)

    new_body["messages"] = new_messages
    return new_body


def turn_temperature(
    span_turn_index: int,
    *,
    total_turns: int,
    protected_recent: int,
    decay: float,
    t0: float,
) -> float:
    """Per-turn initial temperature for the entropy-quench schedule.

    Recent turns (within ``protected_recent`` of the end) keep ``t0``;
    older turns receive ``t0 * decay ** distance_beyond_protected``.
    Smaller temperature means a *more aggressive* schedule because the
    surviving-fraction ``T/T0`` shrinks faster.

    ``decay`` should be in (0, 1].  ``decay == 1`` disables turn-decay.
    """
    if decay >= 1.0 or total_turns <= protected_recent:
        return t0
    distance_from_end = (total_turns - 1) - span_turn_index
    # protected_recent=N means the last N turns (distance 0..N-1) are protected.
    beyond_protected = max(0, distance_from_end - (protected_recent - 1))
    if beyond_protected == 0:
        return t0
    factor = max(0.0, min(1.0, decay**beyond_protected))
    # Floor at t0 * 0.15 — below this, compression destroys task-critical
    # context even when S_E reports adequate fidelity (see paper Sec. 5.3).
    return max(t0 * factor, t0 * 0.15)
