from __future__ import annotations

from typing import Annotated, ClassVar

import pytest

from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.schema import ToolParam, ToolSchema
from myclaw.utils.json_types import JsonObject


class _RepresentativeTool(BaseTool):
    name = "representative"
    description = "Exercise the complete supported Tool declaration surface."
    required = ("text", "empty_text")

    text: Annotated[
        str,
        ToolParam(
            description="Text with explicit constraints.",
            min_length=1,
            max_length=40,
            format="hostname",
        ),
    ]
    empty_text: str
    count: Annotated[
        int | None,
        ToolParam(description="Optional bounded count.", minimum=0, maximum=10),
    ] = None
    enabled: bool = False
    _private: str = "hidden"
    cache: ClassVar[str] = "hidden"

    async def execute(
        self,
        *,
        text: str,
        empty_text: str,
        count: int | None,
        enabled: bool,
    ) -> str:
        return f"{text}:{empty_text}:{count}:{enabled}"


def test_base_tool_generates_complete_openai_function_calling_schema() -> None:
    assert _RepresentativeTool().to_schema() == {
        "type": "function",
        "function": {
            "name": "representative",
            "description": "Exercise the complete supported Tool declaration surface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text with explicit constraints.",
                        "minLength": 1,
                        "maxLength": 40,
                        "format": "hostname",
                    },
                    "empty_text": {"type": "string"},
                    "count": {
                        "type": ["integer", "null"],
                        "description": "Optional bounded count.",
                        "minimum": 0,
                        "maximum": 10,
                        "default": None,
                    },
                    "enabled": {"type": "boolean", "default": False},
                },
                "required": ["text", "empty_text"],
            },
        },
    }


def test_schema_uses_only_direct_public_parameter_annotations() -> None:
    class ParentTool(BaseTool):
        name = "parent"
        description = "Parent Tool."
        inherited: str = "not inherited"

        async def execute(self, *, inherited: str) -> str:
            return inherited

    class ChildTool(ParentTool):
        name = "child"
        description = "Child Tool."
        required = ("direct",)
        direct: int
        metadata: ClassVar[str] = "not a parameter"

        async def execute(self, *, direct: int) -> str:  # type: ignore[override]
            return str(direct)

    properties = ChildTool().to_schema()["function"]
    assert isinstance(properties, dict)
    parameters = properties["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["properties"] == {"direct": {"type": "integer"}}


def test_required_string_is_not_implicitly_nonempty() -> None:
    schema = _RepresentativeTool().to_schema()
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["empty_text"] == {"type": "string"}


def test_schema_exports_are_detached() -> None:
    tool = _RepresentativeTool()
    first = tool.to_schema()
    first_function = first["function"]
    assert isinstance(first_function, dict)
    first_function["name"] = "mutated"
    first_parameters = first_function["parameters"]
    assert isinstance(first_parameters, dict)
    first_required = first_parameters["required"]
    assert isinstance(first_required, list)
    first_required.append("mutated")

    second = tool.to_schema()
    assert second["function"] != first_function
    assert second == ToolSchema.from_tool(_RepresentativeTool).to_openai()


def test_base_tool_rejects_a_schema_override() -> None:
    with pytest.raises(TypeError, match="cannot override"):

        class OverridesSchema(BaseTool):
            name = "override"
            description = "Invalid schema override."

            def to_schema(self) -> JsonObject:  # type: ignore[misc]
                return {}

            async def execute(self) -> str:
                return ""


@pytest.mark.parametrize("retry_count", [True, -1, 6])
def test_base_tool_rejects_invalid_retry_counts(retry_count: object) -> None:
    with pytest.raises(TypeError, match="zero through five"):
        type(
            "InvalidRetryTool",
            (BaseTool,),
            {
                "name": "invalid_retry",
                "description": "Invalid retry count.",
                "max_retries": retry_count,
                "execute": _valid_execute,
            },
        )


async def _valid_execute(self: object, *, value: str) -> str:
    return value


def test_base_tool_rejects_non_async_execution() -> None:
    def execute(self: object, *, value: str) -> str:
        return value

    with pytest.raises(TypeError, match="asynchronous"):
        type(
            "SyncTool",
            (BaseTool,),
            {"name": "sync", "description": "Sync Tool.", "execute": execute},
        )


def test_base_tool_rejects_positional_execution_parameters() -> None:
    async def execute(self: object, value: str) -> str:
        return value

    with pytest.raises(TypeError, match="keyword parameters"):
        type(
            "PositionalTool",
            (BaseTool,),
            {"name": "positional", "description": "Positional Tool.", "execute": execute},
        )


def test_base_tool_rejects_non_string_execution_return() -> None:
    async def execute(self: object) -> int:
        return 1

    with pytest.raises(TypeError, match="string return"):
        type(
            "IntegerResultTool",
            (BaseTool,),
            {"name": "integer_result", "description": "Integer result.", "execute": execute},
        )


def test_tool_schema_rejects_unsupported_parameter_annotations() -> None:
    class UnsupportedTool(BaseTool):
        name = "unsupported"
        description = "Unsupported parameter."
        values: tuple[str, ...] = ()

        async def execute(self, *, values: tuple[str, ...]) -> str:
            return ",".join(values)

    with pytest.raises(TypeError, match="unsupported annotation"):
        UnsupportedTool().to_schema()


def test_tool_error_contains_only_a_public_safe_message() -> None:
    error = ToolError("The path could not be read.")

    assert error.message == "The path could not be read."
    assert str(error) == error.message
    assert not hasattr(error, "code")
