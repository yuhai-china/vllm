# O2 output parser: reasoning (<think>...</think>) + XML tool calls
# (<tool_call><function=name><parameter=k>v</parameter></function></tool_call>)
# in a single ParserEngine, registered under the name "o2" for both the
# reasoning-parser and tool-parser registries.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)


def _o2_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}
    for match in _PARAM_RE.finditer(raw_args):
        params[match.group(1)] = match.group(2).strip()
    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m and m.group(1):
            params[m.group(1)] = m.group(2).strip()
    return json.dumps(params, ensure_ascii=False)


def o2_config(thinking: bool = True) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="o2",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            # -- reasoning --
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING, ()
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT, (EventType.REASONING_END,)
            ),
            # absorb a duplicated </think> after entering CONTENT
            (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT, ()),
            # tool call straight out of reasoning (implicit reasoning end)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # -- tool calls --
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # tolerate <function= without a preceding <tool_call>
            (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME, (EventType.TOOL_CALL_START,)
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.CONTENT, (EventType.TOOL_CALL_END,)
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME, ()
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS, ()
            ),
            (ParserState.TOOL_NAME, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN, (EventType.TOOL_CALL_END,)
            ),
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN, (EventType.TOOL_CALL_END,)
            ),
            (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
                ParserState.TOOL_ARGS, (EventType.ARG_VALUE_CHUNK,)
            ),
            (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
                ParserState.TOOL_ARGS, (EventType.ARG_VALUE_CHUNK,)
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT, ()
            ),
            # consecutive tool call without closing </tool_call>
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE, (EventType.TOOL_CALL_START,)
            ),
            (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME, (EventType.TOOL_CALL_START,)
            ),
        },
        arg_converter=_o2_arg_converter,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
    )


class O2Parser(ParserEngine):
    """O2 parser: ``<think>`` reasoning + XML tool calls in one engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault(
            "parser_engine_config", o2_config(thinking=self.thinking_enabled)
        )
        super().__init__(tokenizer, tools, **kwargs)
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get(TOOL_CALL_START)
        self._tool_call_end_token_id: int | None = vocab.get(TOOL_CALL_END)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if super().is_reasoning_end(input_ids):
            return True
        tool_call_id = self._tool_call_token_id
        tool_call_end_id = self._tool_call_end_token_id
        reasoning_start_id = self._reasoning_start_token_id
        if tool_call_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if (
                    reasoning_start_id is not None
                    and input_ids[i] == reasoning_start_id
                ):
                    return False
                if input_ids[i] == tool_call_id:
                    if tool_call_end_id is not None and any(
                        input_ids[j] == tool_call_end_id
                        for j in range(i + 1, len(input_ids))
                    ):
                        continue
                    return True
        return False


O2ReasoningParser, O2ToolParser = make_adapters(O2Parser)


def register_parsers():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_module(name="o2", module=O2ReasoningParser)
    ToolParserManager.register_module(name="o2", module=O2ToolParser)


__all__ = [
    "O2Parser",
    "O2ReasoningParser",
    "O2ToolParser",
    "register_parsers",
]
