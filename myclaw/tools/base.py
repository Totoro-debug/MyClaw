"""The nominal BaseTool declaration contract."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod, update_abstractmethods
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypedDict,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from myclaw.tools.schema import Schema, ToolParam, parameter
from myclaw.utils.json_types import JsonObject

_METADATA_NAMES = frozenset({"name", "description", "required", "max_retries", "parameters"})
_CLASS_VAR_ORIGIN: object = ClassVar
_MISSING = object()
_INVALID_SCHEMA_MARKER = "__schema_declaration_invalid__"


class OpenAIFunctionSchema(TypedDict):
    """The function member of an OpenAI Function Calling Tool schema."""

    name: str
    description: str
    parameters: JsonObject


class OpenAIToolSchema(TypedDict):
    """An OpenAI Function Calling schema."""

    type: Literal["function"]
    function: OpenAIFunctionSchema


class ToolError(Exception):
    """An expected Tool failure whose message is safe to return to the model."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class BaseTool(ABC):
    """Declare one Tool and expose its model-visible parameter contract."""

    name: ClassVar[str]
    description: ClassVar[str]
    required: ClassVar[tuple[str, ...]] = ()
    max_retries: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "to_schema" in cls.__dict__:
            raise TypeError("Concrete Tools cannot override BaseTool.to_schema()")
        _validate_retry_count(cls)
        execute = cls.__dict__.get("execute")
        if execute is not None:
            _validate_execute(execute)
        if "parameters" not in cls.__dict__:
            declared = _decorated_schema(execute)
            if declared is not None:
                cast(Any, cls).parameters = declared
            else:
                try:
                    legacy = _schema_from_class(cls, execute)
                except (TypeError, ValueError):
                    # Keep declaration errors at the public to_schema boundary for the bridge.
                    if _has_legacy_declaration(cls, execute):
                        cast(Any, cls).parameters = Schema.object({})
                        cast(Any, cls).__schema_declaration_invalid__ = True
                else:
                    cast(Any, cls).parameters = legacy
        update_abstractmethods(cls)

    if TYPE_CHECKING:
        parameters: ClassVar[Schema]
    else:

        @property
        @abstractmethod
        def parameters(self) -> Schema:
            """Return the root object Schema for this Tool's arguments."""
            raise NotImplementedError

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        """Execute one already prepared Tool invocation and return text."""
        raise NotImplementedError

    def prepare(self, arguments: JsonObject) -> JsonObject:
        """Temporary bridge for concrete Tools awaiting the final pipeline migration."""
        return arguments

    def confirmation_finished(self) -> None:
        """Temporary bridge for the current Gateway confirmation lifecycle."""
        return None

    @final
    def to_schema(self) -> OpenAIToolSchema:
        """Generate a detached OpenAI Function Calling schema."""
        tool_type = type(self)
        name = getattr(tool_type, "name", None)
        description = getattr(tool_type, "description", None)
        if not isinstance(name, str) or not name:
            raise TypeError("Tool name must be a non-empty string")
        if not isinstance(description, str) or not description:
            raise TypeError("Tool description must be a non-empty string")
        decorated_schema = getattr(tool_type, "__tool_schema__", None)
        if isinstance(decorated_schema, Schema):
            schema = decorated_schema
        else:
            if getattr(tool_type, _INVALID_SCHEMA_MARKER, False):
                _schema_from_class(tool_type, getattr(tool_type, "execute", None))
            inferred_schema = getattr(tool_type, "parameters", _MISSING)
            if isinstance(inferred_schema, Schema):
                schema = inferred_schema
            else:
                schema = _schema_from_class(tool_type, getattr(tool_type, "execute", None))
        if schema.kind != "object":
            raise TypeError("Tool parameters must use an object root Schema")
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema.to_json_schema(),
            },
        }


def _validate_retry_count(tool_type: type[object]) -> None:
    retries = getattr(tool_type, "max_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
        raise TypeError("Tool max_retries must be an integer from zero through five")


def _has_legacy_declaration(tool_type: type[object], execute: object) -> bool:
    return bool(tool_type.__dict__.get("__annotations__")) or execute is not None


def _validate_execute(execute: object) -> None:
    if not inspect.iscoroutinefunction(execute):
        raise TypeError("Concrete Tool execute() must be asynchronous")
    parameters = tuple(inspect.signature(cast(Any, execute)).parameters.values())
    if (
        not parameters
        or parameters[0].name != "self"
        or parameters[0].kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        or any(parameter.kind is not inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:])
    ):
        raise TypeError("Concrete Tool execute() must accept only self and keyword parameters")
    try:
        return_annotation = get_type_hints(cast(Any, execute)).get("return")
    except (NameError, TypeError) as error:
        raise TypeError(
            "Concrete Tool execute() return annotation could not be resolved"
        ) from error
    if return_annotation is not str:
        raise TypeError("Concrete Tool execute() must declare a string return")


def _decorated_schema(execute: object) -> Schema | None:
    if execute is None:
        return None
    declared = getattr(execute, "__tool_schema__", None)
    return declared if isinstance(declared, Schema) else None


def _schema_from_class(tool_type: type[object], execute: object) -> Schema:
    try:
        annotations = inspect.get_annotations(tool_type, eval_str=True)
    except (NameError, TypeError) as error:
        name = getattr(tool_type, "name", tool_type.__name__)
        raise TypeError(f"Tool annotations for {name} could not be resolved") from error

    required = getattr(tool_type, "required", ())
    if not isinstance(required, tuple) or any(
        not isinstance(item, str) or not item for item in required
    ):
        raise TypeError("Tool required parameters must be a tuple of non-empty strings")
    if len(set(required)) != len(required):
        raise ValueError("Tool required parameters must be unique")

    if not annotations and execute is None:
        raise TypeError("Tool parameters are not declared")
    if not annotations:
        annotations = _execute_annotations(execute)
    properties: dict[str, Schema] = {}
    for parameter_name, annotation in annotations.items():
        if (
            parameter_name.startswith("_")
            or parameter_name in _METADATA_NAMES
            or get_origin(annotation) is _CLASS_VAR_ORIGIN
        ):
            continue
        metadata = _annotation_metadata(parameter_name, annotation)
        base_annotation, nullable = _unwrap_annotation(annotation)
        default = tool_type.__dict__.get(parameter_name, _MISSING)
        properties[parameter_name] = _schema_for_annotation(
            parameter_name,
            base_annotation,
            metadata,
            nullable=nullable,
            default=default,
        )
    return Schema.object(properties, required=required)


def _execute_annotations(execute: object) -> dict[str, object]:
    if execute is None:
        return {}
    try:
        signature = inspect.signature(cast(Any, execute))
    except (TypeError, ValueError) as error:
        raise TypeError("Tool execute() signature could not be inspected") from error
    annotations: dict[str, object] = {}
    for execute_parameter in tuple(signature.parameters.values())[1:]:
        if execute_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        if execute_parameter.annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool parameter {execute_parameter.name} has no annotation")
        annotations[execute_parameter.name] = execute_parameter.annotation
    return annotations


def _annotation_metadata(parameter_name: str, annotation: object) -> ToolParam:
    if get_origin(annotation) is not Annotated:
        return ToolParam()
    _, *extras = get_args(annotation)
    metadata = next((item for item in extras if isinstance(item, ToolParam)), ToolParam())
    if any(not isinstance(item, ToolParam) for item in extras):
        raise TypeError(f"Tool parameter {parameter_name} has unsupported Annotated metadata")
    return metadata


def _unwrap_annotation(annotation: object) -> tuple[object, bool]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in {Union, UnionType}:
        members = get_args(annotation)
        non_none = tuple(member for member in members if member is not NoneType)
        if len(members) != 2 or len(non_none) != 1:
            raise TypeError("Tool parameter has an unsupported union")
        return non_none[0], True
    return annotation, False


def _schema_for_annotation(
    parameter_name: str,
    annotation: object,
    metadata: ToolParam,
    *,
    nullable: bool,
    default: object,
) -> Schema:
    default_kwargs: dict[str, object] = {}
    if default is not _MISSING:
        default_kwargs["default"] = default
    common: dict[str, Any] = {
        "description": metadata.description,
        "nullable": nullable,
        **default_kwargs,
    }
    try:
        if annotation is str:
            return Schema.string(
                **common,
                min_length=metadata.min_length,
                max_length=metadata.max_length,
                format=metadata.format,
                pattern=metadata.pattern,
            )
        if annotation is int:
            if (
                metadata.min_length is not None
                or metadata.max_length is not None
                or metadata.format is not None
            ):
                raise TypeError(
                    f"Tool parameter {parameter_name} has string constraints on a non-string"
                )
            return Schema.integer(
                **common,
                minimum=metadata.minimum,
                maximum=metadata.maximum,
            )
        if annotation is float:
            if (
                metadata.min_length is not None
                or metadata.max_length is not None
                or metadata.format is not None
            ):
                raise TypeError(
                    f"Tool parameter {parameter_name} has string constraints on a non-string"
                )
            return Schema.number(
                **common,
                minimum=metadata.minimum,
                maximum=metadata.maximum,
            )
        if annotation is bool:
            if any(
                value is not None
                for value in (
                    metadata.min_length,
                    metadata.max_length,
                    metadata.minimum,
                    metadata.maximum,
                    metadata.format,
                )
            ):
                raise TypeError(f"Tool parameter {parameter_name} has incompatible constraints")
            return Schema.boolean(**common)
    except TypeError:
        raise
    raise TypeError(f"Tool parameter {parameter_name} has an unsupported annotation")


__all__ = [
    "BaseTool",
    "OpenAIFunctionSchema",
    "OpenAIToolSchema",
    "Schema",
    "ToolError",
    "ToolParam",
    "parameter",
]
