"""Annotation-driven Tool schema declarations."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import NoneType, UnionType
from typing import Annotated, ClassVar, Literal, TypedDict, Union, get_args, get_origin

from jsonschema import Draft202012Validator, FormatChecker

from myclaw.utils.json_types import JsonObject, JsonScalar

_METADATA_NAMES = frozenset({"name", "description", "required", "max_retries"})
_SUPPORTED_TYPES = frozenset({str, int, bool})
_CLASS_VAR_ORIGIN: object = ClassVar


class OpenAIFunctionSchema(TypedDict):
    """The function member of an OpenAI Function Calling Tool schema."""

    name: str
    description: str
    parameters: JsonObject


class OpenAIToolSchema(TypedDict):
    """A typed OpenAI Function Calling schema snapshot."""

    type: Literal["function"]
    function: OpenAIFunctionSchema


@dataclass(frozen=True, slots=True)
class ToolParam:
    """Optional description and JSON Schema constraints for one Tool parameter."""

    description: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    format: str | None = None

    def __post_init__(self) -> None:
        if self.description is not None and not isinstance(self.description, str):
            msg = "Tool parameter description must be a string"
            raise TypeError(msg)
        if self.format is not None and not isinstance(self.format, str):
            msg = "Tool parameter format must be a string"
            raise TypeError(msg)
        for name, value in (
            ("min_length", self.min_length),
            ("max_length", self.max_length),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                msg = f"Tool parameter {name} must be a nonnegative integer"
                raise ValueError(msg)
        for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                msg = f"Tool parameter {name} must be an integer"
                raise TypeError(msg)
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            msg = "Tool parameter min_length cannot exceed max_length"
            raise ValueError(msg)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            msg = "Tool parameter minimum cannot exceed maximum"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _Parameter:
    name: str
    json_type: Literal["string", "integer", "boolean"]
    nullable: bool
    metadata: ToolParam
    has_default: bool
    default: JsonScalar

    def to_json_schema(self) -> JsonObject:
        schema: JsonObject = {
            "type": [self.json_type, "null"] if self.nullable else self.json_type,
        }
        if self.metadata.description is not None:
            schema["description"] = self.metadata.description
        if self.metadata.min_length is not None:
            schema["minLength"] = self.metadata.min_length
        if self.metadata.max_length is not None:
            schema["maxLength"] = self.metadata.max_length
        if self.metadata.minimum is not None:
            schema["minimum"] = self.metadata.minimum
        if self.metadata.maximum is not None:
            schema["maximum"] = self.metadata.maximum
        if self.metadata.format is not None:
            schema["format"] = self.metadata.format
        if self.has_default:
            schema["default"] = self.default
        return schema


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """An immutable parsed Tool declaration that exports detached schemas."""

    name: str
    description: str
    required: tuple[str, ...]
    parameters: tuple[_Parameter, ...]

    @classmethod
    def from_tool(cls, tool_type: type[object]) -> ToolSchema:
        """Parse the direct model-visible declarations on a concrete Tool class."""
        name = getattr(tool_type, "name", None)
        description = getattr(tool_type, "description", None)
        required = getattr(tool_type, "required", ())
        if not isinstance(name, str) or not name:
            msg = "Tool name must be a non-empty string"
            raise TypeError(msg)
        if not isinstance(description, str) or not description:
            msg = "Tool description must be a non-empty string"
            raise TypeError(msg)
        if not isinstance(required, tuple) or any(
            not isinstance(item, str) or not item for item in required
        ):
            msg = "Tool required parameters must be a tuple of non-empty strings"
            raise TypeError(msg)
        if len(set(required)) != len(required):
            msg = "Tool required parameters must be unique"
            raise ValueError(msg)

        try:
            annotations = inspect.get_annotations(tool_type, eval_str=True)
        except (NameError, TypeError) as exc:
            msg = f"Tool annotations for {name} could not be resolved"
            raise TypeError(msg) from exc

        parameters: list[_Parameter] = []
        for parameter_name, annotation in annotations.items():
            if (
                parameter_name.startswith("_")
                or parameter_name in _METADATA_NAMES
                or get_origin(annotation) is _CLASS_VAR_ORIGIN
            ):
                continue
            json_type, nullable, metadata = _parse_annotation(parameter_name, annotation)
            has_default = parameter_name in tool_type.__dict__
            default = tool_type.__dict__.get(parameter_name)
            is_required = parameter_name in required
            if is_required and has_default:
                msg = f"Required Tool parameter {parameter_name} cannot define a default"
                raise TypeError(msg)
            if not is_required and not has_default:
                msg = f"Optional Tool parameter {parameter_name} must define a default"
                raise TypeError(msg)
            parameter = _Parameter(
                name=parameter_name,
                json_type=json_type,
                nullable=nullable,
                metadata=metadata,
                has_default=has_default,
                default=default,
            )
            _validate_parameter(parameter)
            parameters.append(parameter)

        parameter_names = {parameter.name for parameter in parameters}
        unknown_required = set(required) - parameter_names
        if unknown_required:
            names = ", ".join(sorted(unknown_required))
            msg = f"Required Tool parameters are not declared: {names}"
            raise TypeError(msg)
        return cls(
            name=name,
            description=description,
            required=required,
            parameters=tuple(parameters),
        )

    def to_openai(self) -> OpenAIToolSchema:
        """Return a detached OpenAI Function Calling schema."""
        properties: JsonObject = {
            parameter.name: parameter.to_json_schema() for parameter in self.parameters
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(self.required),
                },
            },
        }


def _parse_annotation(
    parameter_name: str,
    annotation: object,
) -> tuple[Literal["string", "integer", "boolean"], bool, ToolParam]:
    metadata = ToolParam()
    if get_origin(annotation) is Annotated:
        annotation, *annotation_metadata = get_args(annotation)
        tool_metadata = [item for item in annotation_metadata if isinstance(item, ToolParam)]
        if len(tool_metadata) != 1 or len(annotation_metadata) != 1:
            msg = f"Tool parameter {parameter_name} has unsupported Annotated metadata"
            raise TypeError(msg)
        metadata = tool_metadata[0]

    nullable = False
    if get_origin(annotation) in {Union, UnionType}:
        members = get_args(annotation)
        non_none = tuple(member for member in members if member is not NoneType)
        if len(members) == 2 and len(non_none) == 1:
            annotation = non_none[0]
            nullable = True
        else:
            msg = f"Tool parameter {parameter_name} has an unsupported union"
            raise TypeError(msg)

    if annotation not in _SUPPORTED_TYPES:
        msg = f"Tool parameter {parameter_name} has an unsupported annotation"
        raise TypeError(msg)
    json_type: Literal["string", "integer", "boolean"]
    if annotation is str:
        json_type = "string"
    elif annotation is int:
        json_type = "integer"
    elif annotation is bool:
        json_type = "boolean"
    else:
        raise AssertionError("supported Tool annotation did not resolve to a type")
    if json_type != "string" and (
        metadata.min_length is not None
        or metadata.max_length is not None
        or metadata.format is not None
    ):
        msg = f"Tool parameter {parameter_name} has string constraints on a non-string"
        raise TypeError(msg)
    if json_type != "integer" and (
        metadata.minimum is not None or metadata.maximum is not None
    ):
        msg = f"Tool parameter {parameter_name} has numeric constraints on a non-integer"
        raise TypeError(msg)
    return json_type, nullable, metadata


def _validate_parameter(parameter: _Parameter) -> None:
    schema = parameter.to_json_schema()
    Draft202012Validator.check_schema(schema)
    if parameter.has_default and not Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).is_valid(parameter.default):
        msg = f"Default for Tool parameter {parameter.name} does not match its annotation"
        raise TypeError(msg)
