"""Locate instruction and post-instruction token positions in chat prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptTokenPositions:
    """Tokenized prompt plus the two causal readout positions."""

    rendered: str
    input_ids: tuple[int, ...]
    instruction_index: int
    post_instruction_index: int


def instruction_index_from_offsets(
    offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    instruction_start: int,
    instruction_end: int,
) -> int:
    """Return the last token overlapping the user instruction's character span."""

    if instruction_start < 0 or instruction_end <= instruction_start:
        raise ValueError("instruction span must be non-empty and ordered")
    overlapping = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < instruction_end and end > instruction_start
    ]
    if not overlapping:
        raise ValueError("no token overlaps the user instruction")
    return overlapping[-1]


def locate_prompt_positions(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> PromptTokenPositions:
    """Render a chat prompt and locate ``t_inst`` and ``t_post-inst``.

    ``t_inst`` is found with tokenizer offsets rather than a hard-coded number of
    chat-template suffix tokens. This remains correct when a model changes its
    assistant header or thinking-control tokens.
    """

    if not prompt:
        raise ValueError("prompt must not be empty")
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(template_kwargs or {}),
    )
    instruction_start = rendered.rfind(prompt)
    if instruction_start < 0:
        raise ValueError("chat template transformed the prompt; cannot locate it safely")
    encoded = tokenizer(rendered, return_offsets_mapping=True)
    raw_ids = encoded["input_ids"]
    raw_offsets = encoded["offset_mapping"]
    if raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
        raw_offsets = raw_offsets[0]
    input_ids = tuple(int(token_id) for token_id in raw_ids)
    offsets = [(int(start), int(end)) for start, end in raw_offsets]
    if not input_ids:
        raise ValueError("tokenizer returned an empty prompt")
    instruction_index = instruction_index_from_offsets(
        offsets,
        instruction_start,
        instruction_start + len(prompt),
    )
    return PromptTokenPositions(
        rendered=rendered,
        input_ids=input_ids,
        instruction_index=instruction_index,
        post_instruction_index=len(input_ids) - 1,
    )
