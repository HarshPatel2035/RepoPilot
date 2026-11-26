"""Scripted chat model: replays fixed responses so agent tests run offline and deterministically."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class CallLog:
    """Shared, mutable record of model calls (pydantic copies plain lists on model_copy)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    log: Any = None  # CallLog
    bind_kwargs: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self.model_copy(update={"bind_kwargs": kwargs})

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **_: Any
    ) -> ChatResult:
        if self.log is not None:
            self.log.calls.append({"bind_kwargs": self.bind_kwargs, "messages": list(messages)})
        if not self.responses:
            raise AssertionError("scripted model ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name: str, call_id: str = "call-1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
