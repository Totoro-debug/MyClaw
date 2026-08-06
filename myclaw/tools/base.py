"""Nominal base class for annotation-driven Tool capabilities."""

from __future__ import annotations

import inspect
from typing import ClassVar, final, get_type_hints

from myclaw.tools.schema import OpenAIToolSchema, ToolSchema
from myclaw.utils.json_types import JsonObject


class BaseTool:
    """Generate one Tool schema and enforce the concrete execution shape."""

    name: ClassVar[str]
    description: ClassVar[str]
    required: ClassVar[tuple[str, ...]] = ()
    max_retries: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "to_schema" in cls.__dict__:
            msg = "Concrete Tools cannot override BaseTool.to_schema()"
            raise TypeError(msg)
        retries = getattr(cls, "max_retries", 0)
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
            msg = "Tool max_retries must be an integer from zero through five"
            raise TypeError(msg)
        execute = cls.__dict__.get("execute")
        if execute is None or not inspect.iscoroutinefunction(execute):
            msg = "Concrete Tool execute() must be asynchronous"
            raise TypeError(msg)
        parameters = tuple(inspect.signature(execute).parameters.values())
        if (
            not parameters
            or parameters[0].name != "self"
            or parameters[0].kind
            not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            or any(
                parameter.kind is not inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:]
            )
        ):
            msg = "Concrete Tool execute() must accept only self and keyword parameters"
            raise TypeError(msg)
        try:
            return_annotation = get_type_hints(execute).get("return")
        except (NameError, TypeError) as exc:
            msg = "Concrete Tool execute() return annotation could not be resolved"
            raise TypeError(msg) from exc
        if return_annotation is not str:
            msg = "Concrete Tool execute() must declare a string return"
            raise TypeError(msg)

    def prepare(self, arguments: JsonObject) -> JsonObject:
        """Select the effective declared arguments before coercion and validation."""
        return arguments

    @final
    def to_schema(self) -> OpenAIToolSchema:
        """Generate a detached OpenAI Function Calling schema."""
        return ToolSchema.from_tool(type(self)).to_openai()
