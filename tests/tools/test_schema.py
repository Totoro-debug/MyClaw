from __future__ import annotations

from typing import cast

import pytest

from myclaw.tools.schema import Schema, SchemaError
from myclaw.utils.json_types import JsonObject, JsonValue


def test_schema_builders_export_restricted_detached_json_schema() -> None:
    schema = Schema.object(
        {
            "name": Schema.string(description="A name", min_length=1),
            "count": Schema.integer(default=2, minimum=0),
            "ratio": Schema.number(nullable=True, maximum=1.0),
            "enabled": Schema.boolean(),
            "tags": Schema.array(Schema.string(), min_items=1),
        },
        required=("name", "count", "enabled", "tags"),
        description="Payload",
    )

    first = schema.to_json_schema()
    first_properties = cast(JsonObject, first["properties"])
    cast(JsonObject, first_properties["name"])["description"] = "mutated"
    cast(list[JsonValue], first["required"]).append("other")

    assert schema.to_json_schema() == {
        "type": "object",
        "description": "Payload",
        "properties": {
            "name": {"type": "string", "description": "A name", "minLength": 1},
            "count": {"type": "integer", "default": 2, "minimum": 0},
            "ratio": {"type": ["number", "null"], "maximum": 1.0},
            "enabled": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["name", "count", "enabled", "tags"],
    }


def test_schema_builders_reject_non_json_defaults_and_enum_values() -> None:
    with pytest.raises(TypeError, match=r"Schema default\.invalid must be JSON-compatible"):
        Schema.object(default={"invalid": object()})
    with pytest.raises(ValueError, match=r"Schema enum\[0\] must be JSON-compatible"):
        Schema.number(enum=[float("nan")])


def test_schema_cast_is_recursive_safe_and_does_not_apply_defaults() -> None:
    schema = Schema.object(
        {
            "count": Schema.integer(default=10),
            "enabled": Schema.boolean(),
            "values": Schema.array(Schema.integer()),
            "nested": Schema.object({"ratio": Schema.number()}),
        }
    )

    cast = schema.cast(
        {
            "count": "-2",
            "enabled": "TrUe",
            "values": ["1", 2.0],
            "nested": {"ratio": 1},
            "unknown": {"kept": True},
        }
    )

    assert cast == {
        "count": -2,
        "enabled": True,
        "values": [1, 2],
        "nested": {"ratio": 1},
        "unknown": {"kept": True},
    }
    assert schema.cast({}) == {}
    assert schema.cast({"count": True}) == {"count": True}


def test_schema_validation_aggregates_stable_nested_paths() -> None:
    schema = Schema.object(
        {
            "name": Schema.string(min_length=3),
            "items": Schema.array(
                Schema.object(
                    {
                        "count": Schema.integer(minimum=1),
                        "enabled": Schema.boolean(),
                    },
                    required=("count", "enabled"),
                )
            ),
        },
        required=("name", "items"),
    )

    errors = schema.validate(
        {
            "name": "x",
            "items": [{"count": 0}, {"count": "bad", "enabled": "maybe"}],
        }
    )

    assert [error.path for error in errors] == [
        "$.name",
        "$.items[0].count",
        "$.items[0].enabled",
        "$.items[1].count",
        "$.items[1].enabled",
    ]
    assert all(isinstance(error, SchemaError) for error in errors)
    assert "$.items[1].enabled" in str(errors[-1])


def test_schema_rejects_boolean_as_numeric_and_only_safe_conversions_apply() -> None:
    integer = Schema.integer()
    number = Schema.number()

    assert integer.cast(2.0) == 2
    assert integer.cast("2") == 2
    assert integer.cast(True) is True
    assert number.cast("2") == "2"
    assert integer.validate(True) == [SchemaError("$", "expected integer, got boolean", "type")]
    assert number.validate(True) == [SchemaError("$", "expected number, got boolean", "type")]


def test_schema_validation_matches_emitted_enum_and_unique_item_semantics() -> None:
    nullable_enum = Schema.number(nullable=True, enum=[1])

    assert nullable_enum.validate(1.0) == []
    assert nullable_enum.validate(None) == [
        SchemaError("$", "value is not one of the allowed values", "enum")
    ]
    assert Schema.array(Schema.number(), unique_items=True).validate([1, 1.0]) == [
        SchemaError("$", "must contain unique items", "uniqueItems")
    ]
    assert Schema.array(Schema.object({"value": Schema.number()}), unique_items=True).validate(
        [{"value": 1}, {"value": 1.0}]
    ) == [SchemaError("$", "must contain unique items", "uniqueItems")]


def test_schema_casts_and_validates_declared_additional_properties() -> None:
    additional_integer = Schema.object(
        {"known": Schema.integer()},
        additional_properties=Schema.integer(minimum=0),
    )

    assert additional_integer.cast({"known": "1", "extra": "2"}) == {
        "known": 1,
        "extra": 2,
    }
    assert additional_integer.validate({"known": 1, "extra": -1}) == [
        SchemaError("$.extra", "must be greater than or equal to 0", "minimum")
    ]
    assert Schema.object({"known": Schema.integer()}, additional_properties=False).validate(
        {"known": 1, "extra": 2}
    ) == [SchemaError("$.extra", "additional property is not allowed", "additionalProperties")]
    assert Schema.object({"known": Schema.integer()}).validate({"known": 1, "extra": 2}) == []


def test_schema_multiple_of_is_exact_and_total_for_finite_numbers() -> None:
    assert Schema.number(multiple_of=0.1).validate(0.3) == []
    assert Schema.number(multiple_of=1).validate(1e-13) == [
        SchemaError("$", "must be a multiple of 1", "multipleOf")
    ]
    assert Schema.number(multiple_of=5e-324).validate(1.0) == []
