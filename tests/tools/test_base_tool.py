from __future__ import annotations

import inspect
from typing import Annotated, Any, ClassVar, cast

import pytest

from myclaw.tools.base import BaseTool, ToolError, ToolParam
from myclaw.tools.schema import Object, String, parameter


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
    assert second == tool.to_schema()


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


def test_base_tool_is_abstract_and_parameter_decorator_injects_root_schema() -> None:
    declared = Object({"text": String()}, required=("text",))

    class DecoratedTool(BaseTool):
        name = "decorated"
        description = "A decorated Tool."

        @parameter(declared)
        async def execute(self, *, text: str) -> str:
            return text

    assert inspect.isabstract(BaseTool)
    assert not inspect.isabstract(DecoratedTool)
    assert DecoratedTool.parameters == declared
    assert DecoratedTool().to_schema()["function"]["parameters"] == declared.to_json_schema()


def test_tool_without_a_declaration_or_execution_remains_abstract() -> None:
    class IncompleteTool(BaseTool):
        name = "incomplete"
        description = "An incomplete Tool."

    assert inspect.isabstract(IncompleteTool)


def test_tool_with_parameters_but_without_execution_remains_abstract() -> None:
    class ParametersOnlyTool(BaseTool):
        name = "parameters_only"
        description = "A Tool without execution."
        value: str

    assert inspect.isabstract(ParametersOnlyTool)
    with pytest.raises(TypeError, match="abstract method 'execute'"):
        cast(Any, ParametersOnlyTool)()


def test_class_parameter_decorator_replaces_failed_legacy_inference() -> None:
    declared = Object({"value": String()}, required=("value",))

    @parameter(declared)
    class DecoratedTool(BaseTool):
        name = "class_decorated"
        description = "A class-decorated Tool."
        value: Annotated[str, "legacy metadata is irrelevant"]

        async def execute(self, *, value: str) -> str:
            return value

    assert not inspect.isabstract(DecoratedTool)
    assert DecoratedTool().to_schema() == {
        "type": "function",
        "function": {
            "name": "class_decorated",
            "description": "A class-decorated Tool.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
