"""Tests for the structural multi-turn compression planner."""

from __future__ import annotations

from entropy_gate.structure import (
    apply_compression,
    detect_api,
    plan_compression,
    turn_temperature,
)


def test_detect_anthropic_via_system_string() -> None:
    body = {"system": "you are helpful", "messages": []}
    assert detect_api(body) == "anthropic"


def test_detect_anthropic_via_tool_use_block() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                ],
            }
        ]
    }
    assert detect_api(body) == "anthropic"


def test_detect_openai_via_system_role() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "hi"},
        ]
    }
    assert detect_api(body) == "openai"


def test_detect_openai_default_for_string_content() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert detect_api(body) == "openai"


def test_plan_skips_last_user_message() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "first user message " * 20},
            {"role": "assistant", "content": "assistant reply " * 20},
            {"role": "user", "content": "the live query " * 20},
        ]
    }
    plan = plan_compression(body, min_chars=10)
    # The last user message must be frozen — no span pointing at index 2.
    indices = [s.message_index for s in plan.spans]
    assert 2 not in indices
    assert 0 in indices
    assert 1 in indices


def test_plan_preserves_anthropic_tool_blocks() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me check " * 20},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "search",
                        "input": {"q": "x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "the result",
                    }
                ],
            },
            {"role": "user", "content": "final query " * 20},
        ]
    }
    plan = plan_compression(body, min_chars=10)
    # The text block in the assistant message at idx 0, block 0 IS compressible.
    # But the message at idx 0 ALSO has a tool_use → whole message is frozen.
    span_keys = {(s.message_index, s.block_index) for s in plan.spans}
    assert (0, 0) not in span_keys
    # The tool_result message (idx 1) is frozen.
    assert all(s.message_index != 1 for s in plan.spans)
    # The last user (idx 2) is frozen.
    assert all(s.message_index != 2 for s in plan.spans)


def test_plan_preserves_openai_tool_messages() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "do a thing " * 20},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "tool output " * 20},
            {"role": "user", "content": "final query " * 20},
        ]
    }
    plan = plan_compression(body, min_chars=10)
    indices = {s.message_index for s in plan.spans}
    # idx 0 (user) is compressible — not the last user.
    assert 0 in indices
    # idx 1 (assistant with tool_calls) — frozen.
    assert 1 not in indices
    # idx 2 (role=tool) — frozen.
    assert 2 not in indices
    # idx 3 (last user) — frozen.
    assert 3 not in indices


def test_apply_compression_writes_back_string_content() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "long original message " * 20},
            {"role": "user", "content": "final"},
        ]
    }
    plan = plan_compression(body, min_chars=10)
    repls = {(s.message_index, s.block_index): "short" for s in plan.spans}
    new_body = apply_compression(body, plan, repls)
    assert new_body["messages"][0]["content"] == "short"
    # Original unchanged
    assert body["messages"][0]["content"].startswith("long original")
    # Last user untouched.
    assert new_body["messages"][1]["content"] == "final"


def test_apply_compression_writes_back_anthropic_block() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "long block " * 30},
                ],
            },
            {"role": "user", "content": "final query " * 20},
        ]
    }
    plan = plan_compression(body, min_chars=10)
    repls = {(s.message_index, s.block_index): "short" for s in plan.spans}
    new_body = apply_compression(body, plan, repls)
    block = new_body["messages"][0]["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "short"


def test_turn_temperature_protects_recent() -> None:
    t0 = 1.0
    # protected_recent=2, total=5: turns 4 and 3 keep t0; older shrink.
    assert turn_temperature(4, total_turns=5, protected_recent=2, decay=0.5, t0=t0) == t0
    assert turn_temperature(3, total_turns=5, protected_recent=2, decay=0.5, t0=t0) == t0
    # Turn 2 is 1 beyond protected → t0 * 0.5
    assert turn_temperature(2, total_turns=5, protected_recent=2, decay=0.5, t0=t0) == 0.5
    # Turn 0 is 3 beyond protected → t0 * 0.125 but floored at t0*0.05
    val = turn_temperature(0, total_turns=5, protected_recent=2, decay=0.5, t0=t0)
    assert val == 0.125


def test_turn_temperature_decay_one_is_noop() -> None:
    assert turn_temperature(0, total_turns=10, protected_recent=2, decay=1.0, t0=1.0) == 1.0
