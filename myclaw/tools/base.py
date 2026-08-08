"""Annotation-driven Tool declarations."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import NoneType, UnionType
from typing import Annotated, ClassVar, Literal, TypedDict, Union, cast, final, get_args, get_origin

from myclaw.utils.json_types import JsonObject, JsonScalar

_METADATA_NAMES = frozenset({"name", "description", "required", "max_retries"})
_JSON_TYPES: dict[object, str] = {str: "string", int: "integer", bool: "boolean"}
_CLASS_VAR_ORIGIN: object = ClassVar


class OpenAIFunctionSchema(TypedDict):
    """The function member of an OpenAI Function Calling Tool schema."""

    name: str
    description: str
    parameters: JsonObject


class OpenAIToolSchema(TypedDict):
    """An OpenAI Function Calling schema."""

    type: Literal["function"]
    function: OpenAIFunctionSchema


@dataclass(frozen=True, slots=True)
class ToolParam:
    """Optional JSON Schema metadata for one Tool parameter."""

    description: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    format: str | None = None


class ToolError(Exception):
    """An expected Tool failure whose message is safe to return to the model."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class BaseTool:
    """Declare one annotation-driven Tool capability."""

    name: ClassVar[str]
    description: ClassVar[str]
    required: ClassVar[tuple[str, ...]] = ()
    max_retries: ClassVar[int] = 0

    def prepare(self, arguments: JsonObject) -> JsonObject:
        """Select the declared arguments used by this invocation."""
        return arguments

    def confirmation_finished(self) -> None:
        """Release invocation-local confirmation state."""

    @final
    def to_schema(self) -> OpenAIToolSchema:
        """Generate a detached OpenAI Function Calling schema."""
        tool_type = type(self)
        properties: JsonObject = {}
        for name, annotation in inspect.get_annotations(tool_type, eval_str=True).items():
            if (
                name.startswith("_")
                or name in _METADATA_NAMES
                or get_origin(annotation) is _CLASS_VAR_ORIGIN
            ):
                continue
            schema = _parameter_schema(name, annotation)
            if name in tool_type.__dict__:
                schema["default"] = cast(JsonScalar, tool_type.__dict__[name])
            properties[name] = schema
        return {
            "type": "function",
            "function": {
                "name": tool_type.name,
                "description": tool_type.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tool_type.required),
                },
            },
        }


def _parameter_schema(name: str, annotation: object) -> JsonObject:
    metadata = ToolParam()
    if get_origin(annotation) is Annotated:
        annotation, *extras = get_args(annotation)
        metadata = next((item for item in extras if isinstance(item, ToolParam)), metadata)

    nullable = False
    if get_origin(annotation) in {Union, UnionType}:
        members = get_args(annotation)
        non_none = tuple(member for member in members if member is not NoneType)
        if len(members) != 2 or len(non_none) != 1:
            raise TypeError(f"Tool parameter {name} has an unsupported union")
        annotation = non_none[0]
        nullable = True

    try:
        json_type = _JSON_TYPES[annotation]
    except (KeyError, TypeError) as error:
        raise TypeError(f"Tool parameter {name} has an unsupported annotation") from error

    schema: JsonObject = {"type": [json_type, "null"] if nullable else json_type}
    for field, key in (
        ("description", "description"),
        ("min_length", "minLength"),
        ("max_length", "maxLength"),
        ("minimum", "minimum"),
        ("maximum", "maximum"),
        ("format", "format"),
    ):
        value = getattr(metadata, field)
        if value is not None:
            schema[key] = value
    return schema
