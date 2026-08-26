import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.management.commands import MANAGEMENT_COMMANDS
from myclaw.skills.catalog import SkillUnavailableError, discover_skills
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures.diagnostic_capture import capture_diagnostics


def test_missing_skills_root_is_empty_without_creating_directory(agent_home: Path) -> None:
    home = AgentHome(agent_home)

    catalog = discover_skills(
        agent_home=home,
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.root == (agent_home / "skills").resolve()
    assert catalog.entries == ()
    assert not home.skills_directory.exists()


def test_existing_empty_skills_root_is_empty(agent_home: Path) -> None:
    root = agent_home / "skills"
    root.mkdir(parents=True)

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.root == root.resolve()
    assert catalog.entries == ()
    assert root.is_dir()


def test_valid_direct_child_retains_trimmed_metadata_only(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        '---\nname: "  plan  "\ndescription: "  Do useful work.  "\n---\nsecret body\n',
        encoding="utf-8",
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert len(catalog.entries) == 1
    entry = catalog.entries[0]
    assert entry.metadata.name == "plan"
    assert entry.metadata.description == "Do useful work."
    assert entry.metadata.path == instruction.resolve()
    assert entry.always_body is None
    assert catalog.get("plan") == entry


def test_name_starting_with_a_digit_is_excluded(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "numeric" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: 1plan\ndescription: A plan\n---\n",
        encoding="utf-8",
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.entries == ()


def test_reserved_management_command_name_is_excluded(agent_home: Path) -> None:
    for command in MANAGEMENT_COMMANDS:
        name = command.token.removeprefix("/")
        instruction = agent_home / "skills" / f"{name}-skill" / "SKILL.md"
        instruction.parent.mkdir(parents=True)
        instruction.write_text(
            f"---\nname: {name}\ndescription: Reserved command guide\n---\n",
            encoding="utf-8",
        )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
        enable_always_load=False,
    )

    assert catalog.entries == ()


def test_first_valid_duplicate_in_canonical_path_order_wins(agent_home: Path) -> None:
    later = agent_home / "skills" / "z-candidate" / "SKILL.md"
    earlier = agent_home / "skills" / "a-candidate" / "SKILL.md"
    for instruction in (later, earlier):
        instruction.parent.mkdir(parents=True)
        instruction.write_text(
            "---\nname: duplicate\ndescription: Same name\n---\n",
            encoding="utf-8",
        )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert tuple(entry.metadata.path for entry in catalog.entries) == (earlier.resolve(),)


def test_valid_entries_follow_canonical_path_order_regardless_of_creation_order(
    agent_home: Path,
) -> None:
    later = agent_home / "skills" / "z-candidate" / "SKILL.md"
    earlier = agent_home / "skills" / "a-candidate" / "SKILL.md"
    for instruction, name in ((later, "later"), (earlier, "earlier")):
        instruction.parent.mkdir(parents=True)
        instruction.write_text(
            f"---\nname: {name}\ndescription: Valid metadata\n---\n",
            encoding="utf-8",
        )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert tuple(entry.metadata.name for entry in catalog.entries) == ("earlier", "later")
    assert tuple(entry.metadata.path for entry in catalog.entries) == (
        earlier.resolve(),
        later.resolve(),
    )


def test_invalid_candidate_logs_path_and_reason_without_body(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "invalid" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: invalid\ndescription: missing closing delimiter\nsecret: body\n"
        b"SECRET-SKILL-BODY\n"
    )
    diagnostics = capture_diagnostics()

    try:
        catalog = discover_skills(
            agent_home=AgentHome(agent_home),
            reserved_names=(),
            enable_always_load=False,
        )
    finally:
        diagnostics.close()

    assert catalog.entries == ()
    assert "Skipping invalid Skill candidate" in diagnostics.event_text
    assert str(instruction.parent) in diagnostics.event_text
    assert "SECRET-SKILL-BODY" not in diagnostics.text


def test_only_direct_child_skill_directories_are_scanned(agent_home: Path) -> None:
    direct = agent_home / "skills" / "direct" / "SKILL.md"
    nested = agent_home / "skills" / "container" / "nested" / "SKILL.md"
    for instruction, name in ((direct, "direct"), (nested, "nested")):
        instruction.parent.mkdir(parents=True)
        instruction.write_text(
            f"---\nname: {name}\ndescription: Valid metadata\n---\n",
            encoding="utf-8",
        )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert tuple(entry.metadata.name for entry in catalog.entries) == ("direct",)


@pytest.mark.parametrize(
    "document",
    (
        "name: valid\ndescription: no opening delimiter\n",
        "---\nname: valid\ndescription: missing closing\n",
        "---\nname: [broken\ndescription: invalid YAML\n---\n",
        "---\n- item\n---\n",
        "---\ndescription: missing name\n---\n",
        "---\nname: valid\ndescription: [not a string]\n---\n",
    ),
)
def test_invalid_frontmatter_shapes_are_excluded(agent_home: Path, document: str) -> None:
    instruction = agent_home / "skills" / "invalid-frontmatter" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(document, encoding="utf-8")

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.entries == ()


@pytest.mark.parametrize(
    ("name", "accepted"),
    (
        ("a", True),
        ("-a", True),
        ("_a", True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("1a", False),
        ("A", False),
        ("a.b", False),
    ),
)
def test_name_character_and_length_contract(
    agent_home: Path,
    name: str,
    accepted: bool,
) -> None:
    instruction = agent_home / "skills" / "name-boundary" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        f'---\nname: "{name}"\ndescription: "Valid description"\n---\n',
        encoding="utf-8",
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert bool(catalog.entries) is accepted


@pytest.mark.parametrize(
    ("description", "accepted"),
    (
        ("", False),
        ("   ", False),
        ("x", True),
        ("x" * 1024, True),
        ("x" * 1025, False),
    ),
)
def test_description_trimmed_length_contract(
    agent_home: Path,
    description: str,
    accepted: bool,
) -> None:
    instruction = agent_home / "skills" / "description-boundary" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        f'---\nname: "description"\ndescription: "{description}"\n---\n',
        encoding="utf-8",
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert bool(catalog.entries) is accepted


def test_non_utf8_frontmatter_is_excluded(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "non-utf8" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: valid\ndescription: \xff\n---\nBODY\n")

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.entries == ()


def test_non_regular_instruction_path_is_excluded(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "directory-instruction" / "SKILL.md"
    instruction.mkdir(parents=True)

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.entries == ()


def test_metadata_discovery_does_not_decode_body_bytes(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "metadata-only" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: metadata-only\ndescription: Metadata is valid\n---\n\xffBODY\n"
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert tuple(entry.metadata.name for entry in catalog.entries) == ("metadata-only",)
    assert catalog.entries[0].always_body is None


def test_instruction_symlink_escape_is_excluded_when_links_are_available(
    agent_home: Path,
) -> None:
    outside = agent_home.parent / "outside" / "SKILL.md"
    outside.parent.mkdir()
    outside.write_text(
        "---\nname: escaped\ndescription: Outside the Skill root\n---\n",
        encoding="utf-8",
    )
    instruction = agent_home / "skills" / "escaped" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    try:
        instruction.symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file links unavailable: {error}")

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.entries == ()


def test_skill_directory_reparse_escape_is_excluded_when_links_are_available(
    agent_home: Path,
) -> None:
    outside = agent_home.parent / "outside-directory"
    instruction = outside / "SKILL.md"
    instruction.parent.mkdir()
    instruction.write_text(
        "---\nname: escaped\ndescription: Outside the Skill root\n---\n",
        encoding="utf-8",
    )
    linked_directory = agent_home / "skills" / "escaped"
    linked_directory.parent.mkdir(parents=True)
    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(linked_directory), str(outside)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            linked_directory.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError, subprocess.CalledProcessError) as error:
        pytest.skip(f"directory links unavailable: {error}")

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.entries == ()


def test_catalog_snapshot_exposes_immutable_entries_and_lookup(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "immutable" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: immutable\ndescription: Immutable metadata\n---\n",
        encoding="utf-8",
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    entry = catalog.get("immutable")

    assert isinstance(catalog.entries, tuple)
    assert entry is catalog.entries[0]
    with pytest.raises(AttributeError):
        catalog.entries.append(entry)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        entry.metadata.name = "changed"  # type: ignore[misc]


def test_invalid_earlier_duplicate_does_not_block_later_valid_entry(agent_home: Path) -> None:
    invalid = agent_home / "skills" / "a-invalid" / "SKILL.md"
    valid = agent_home / "skills" / "b-valid" / "SKILL.md"
    invalid.parent.mkdir(parents=True)
    valid.parent.mkdir(parents=True)
    invalid.write_text(
        "---\nname: duplicate\ndescription: \n---\n",
        encoding="utf-8",
    )
    valid.write_text(
        "---\nname: duplicate\ndescription: Later valid metadata\n---\n",
        encoding="utf-8",
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert tuple(entry.metadata.path for entry in catalog.entries) == (valid.resolve(),)


def test_disabled_always_load_does_not_interpret_always_metadata(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "always-candidate" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: always-candidate\ndescription: Metadata only\nalways: true\n---\n\xffbody\n"
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert len(catalog.entries) == 1
    assert catalog.entries[0].always_body is None


def test_enabled_boolean_always_freezes_the_complete_body(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: always\ndescription: Always loaded\nalways: true\n---\nComplete body\n"
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=True,
    )

    assert catalog.entries[0].always_body == "Complete body\n"


@pytest.mark.parametrize("always_field", ("always: false\n", ""))
def test_enabled_non_opted_in_skill_remains_metadata_only(
    agent_home: Path,
    always_field: str,
) -> None:
    instruction = agent_home / "skills" / "metadata-only" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        (
            "---\nname: metadata-only\ndescription: Metadata only\n"
            + always_field
            + "---\nnot frozen\n"
        ).encode("utf-8")
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=True,
    )

    assert catalog.entries[0].always_body is None


def test_duplicate_always_candidate_does_not_override_the_first_valid_entry(
    agent_home: Path,
) -> None:
    first = agent_home / "skills" / "a-first" / "SKILL.md"
    duplicate = agent_home / "skills" / "b-duplicate" / "SKILL.md"
    first.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    first.write_bytes(
        b"---\nname: duplicate\ndescription: First candidate\nalways: false\n---\nfirst\n"
    )
    duplicate.write_bytes(
        b"---\nname: duplicate\ndescription: Later candidate\nalways: true\n---\nsecond\n"
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=True,
    )

    assert len(catalog.entries) == 1
    assert catalog.entries[0].metadata.description == "First candidate"
    assert catalog.entries[0].always_body is None


def test_enabled_always_non_utf8_body_fails_closed(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "binary-always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: binary-always\ndescription: Binary body\nalways: true\n---\n\xff"
    )

    with pytest.raises(SkillUnavailableError) as failure:
        discover_skills(
            agent_home=AgentHome(agent_home),
            reserved_names=(),
            enable_always_load=True,
        )

    assert failure.value.error.code == "skill_unavailable"


def test_always_body_is_frozen_in_the_final_catalog(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "frozen" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: frozen\ndescription: Frozen body\nalways: true\n---\nfirst\n"
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=True,
    )
    instruction.write_bytes(
        b"---\nname: frozen\ndescription: Frozen body\nalways: true\n---\nsecond\n"
    )

    assert catalog.entries[0].always_body == "first\n"


def test_always_opt_in_change_during_complete_read_fails_closed(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "changing-always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: changing-always\ndescription: Changing policy\nalways: true\n---\n"
        b"body from the opted-in version\n"
    )
    original_open = cast(Callable[..., Any], Path.open)
    open_count = 0

    def replace_after_metadata(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal open_count
        if path.name == "SKILL.md":
            open_count += 1
            if open_count == 2:
                instruction.write_bytes(
                    b"---\nname: changing-always\ndescription: Changing policy\nalways: false\n---\n"
                    b"body from the changed version\n"
                )
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_after_metadata)

    with pytest.raises(SkillUnavailableError) as failure:
        discover_skills(
            agent_home=AgentHome(agent_home),
            reserved_names=(),
            enable_always_load=True,
        )

    assert failure.value.error.code == "skill_unavailable"


def test_enabled_non_boolean_always_warns_once_and_stays_metadata_only(
    agent_home: Path,
) -> None:
    instruction = agent_home / "skills" / "invalid-always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b'---\nname: invalid-always\ndescription: Invalid policy\nalways: "true"\n---\n'
        b"SECRET BODY\n"
    )
    diagnostics = capture_diagnostics()

    try:
        catalog = discover_skills(
            agent_home=AgentHome(agent_home),
            reserved_names=(),
            enable_always_load=True,
        )
    finally:
        diagnostics.close()

    assert catalog.entries[0].always_body is None
    assert diagnostics.event_text.count("Ignoring non-boolean Skill always field") == 1
    assert "SECRET BODY" not in diagnostics.text


def test_read_body_returns_the_complete_current_skill_body(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "reader" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: reader\ndescription: Read the body\n---\nfirst line\nsecond line\n"
    )

    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.read_body(catalog.entries[0].metadata) == "first line\nsecond line\n"


def test_resolve_manual_returns_the_complete_body_for_an_exact_slash_name(
    agent_home: Path,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: planner\ndescription: Plan work\n---\nFollow the plan.\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    invocation = catalog.resolve_manual("/planner")

    assert invocation is not None
    assert invocation.metadata == catalog.entries[0].metadata
    assert invocation.request == ""
    assert invocation.body == "Follow the plan.\n"


@pytest.mark.parametrize(
    "delimiter",
    (" ", "\t", "\r", "\n", "\u2003"),
)
def test_resolve_manual_removes_only_the_first_unicode_whitespace_delimiter(
    agent_home: Path,
    delimiter: str,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nFollow the plan.\n",
        encoding="utf-8",
    )
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    request = f"{delimiter}  keep\n\tthis"

    invocation = catalog.resolve_manual(f"/planner{request}")

    assert invocation is not None
    assert invocation.request == "  keep\n\tthis"


def test_resolve_manual_reads_a_matching_body_once(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nFollow the plan.\n",
        encoding="utf-8",
    )
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    reads: list[object] = []
    original_read_body = catalog.read_body

    def read_body(metadata: object) -> str:
        reads.append(metadata)
        return original_read_body(metadata)  # type: ignore[arg-type]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(catalog, "read_body", read_body)
    try:
        invocation = catalog.resolve_manual("/planner do it")
    finally:
        monkeypatch.undo()

    assert invocation is not None
    assert len(reads) == 1


@pytest.mark.parametrize(
    "raw_input",
    ("planner", " /planner", "/Planner", "/plan", "/unknown", "/config"),
)
def test_resolve_manual_does_not_read_non_matching_input(
    agent_home: Path,
    raw_input: str,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nFollow the plan.\n",
        encoding="utf-8",
    )
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=("/config",),
        enable_always_load=False,
    )
    reads: list[object] = []

    def read_body(metadata: object) -> str:
        reads.append(metadata)
        return "unexpected"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(catalog, "read_body", read_body)
    try:
        assert catalog.resolve_manual(raw_input) is None
    finally:
        monkeypatch.undo()

    assert reads == []


def test_resolve_manual_rejects_non_string_input(agent_home: Path) -> None:
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    with pytest.raises(TypeError, match="Skill input must be a string"):
        catalog.resolve_manual(cast(str, object()))


def test_read_body_rejects_metadata_changed_after_discovery(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "stale" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: stale\ndescription: Original\n---\nbody\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    instruction.write_bytes(b"---\nname: stale\ndescription: Changed\n---\nbody\n")

    with pytest.raises(SkillUnavailableError) as failure:
        catalog.read_body(catalog.entries[0].metadata)

    assert failure.value.error.code == "skill_unavailable"
    assert str(failure.value) == failure.value.error.message


def test_read_body_maps_missing_target_to_skill_unavailable(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "missing" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: missing\ndescription: Missing\n---\nbody\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    instruction.unlink()

    with pytest.raises(SkillUnavailableError) as failure:
        catalog.read_body(catalog.entries[0].metadata)

    assert failure.value.error.code == "skill_unavailable"


def test_read_body_maps_unreadable_target_to_skill_unavailable(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "unreadable" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: unreadable\ndescription: Unreadable\n---\nbody\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    def deny_open(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise PermissionError("denied by test")

    monkeypatch.setattr(Path, "open", deny_open)

    with pytest.raises(SkillUnavailableError) as failure:
        catalog.read_body(catalog.entries[0].metadata)

    assert failure.value.error.code == "skill_unavailable"


def test_read_body_maps_non_utf8_content_to_skill_unavailable(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "binary" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: binary\ndescription: Binary\n---\n\xffbody\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    with pytest.raises(SkillUnavailableError) as failure:
        catalog.read_body(catalog.entries[0].metadata)

    assert failure.value.error.code == "skill_unavailable"


def test_read_body_does_not_cache_the_body(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "fresh" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: fresh\ndescription: Fresh\n---\nfirst\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    metadata = catalog.entries[0].metadata
    assert catalog.read_body(metadata) == "first\n"

    instruction.write_bytes(b"---\nname: fresh\ndescription: Fresh\n---\nsecond\n")

    assert catalog.read_body(metadata) == "second\n"


def test_read_body_keeps_regular_hardlinks_readable(agent_home: Path) -> None:
    outside = agent_home.parent / "shared-skill.md"
    outside.write_bytes(b"---\nname: linked\ndescription: Linked body\n---\nbody\n")
    instruction = agent_home / "skills" / "linked" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    try:
        instruction.hardlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"hard links unavailable: {error}")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )

    assert catalog.read_body(catalog.entries[0].metadata) == "body\n"


def test_read_body_uses_one_opened_descriptor_instead_of_a_second_path_read(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "stable" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: stable\ndescription: Stable body\n---\noriginal\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    path_reads: list[Path] = []

    def substitute_path_read(path: Path) -> bytes:
        path_reads.append(path)
        return b"---\nname: stable\ndescription: Stable body\n---\noutside\n"

    monkeypatch.setattr(Path, "read_bytes", substitute_path_read)

    assert catalog.read_body(catalog.entries[0].metadata) == "original\n"
    assert path_reads == []


def test_read_body_rejects_a_symlink_replacement_after_open(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "stable" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: stable\ndescription: Stable body\n---\noriginal\n")
    outside = agent_home.parent / "replacement.md"
    outside.write_bytes(b"---\nname: stable\ndescription: Stable body\n---\noutside\n")
    probe = instruction.parent / "link-probe"
    try:
        probe.symlink_to(outside)
        probe.unlink()
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file links unavailable: {error}")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    validate_opened = HOST_FILESYSTEM.require_opened_contained_regular_file

    def replace_before_validation(descriptor: int, path: Path, *, within: Path) -> Path:
        instruction.unlink()
        instruction.symlink_to(outside)
        return validate_opened(descriptor, path, within=within)

    monkeypatch.setattr(
        HOST_FILESYSTEM,
        "require_opened_contained_regular_file",
        replace_before_validation,
    )

    with pytest.raises(SkillUnavailableError) as failure:
        catalog.read_body(catalog.entries[0].metadata)

    assert failure.value.error.code == "skill_unavailable"


@pytest.mark.parametrize(
    "replacement",
    (
        b"---\nname: 1-invalid\ndescription: Stable body\n---\nbody\n",
        b"---\nname: stable\ndescription: ''\n---\nbody\n",
        b"---\n- name\n- stable\n---\nbody\n",
    ),
)
def test_read_body_uses_the_same_metadata_rules_as_discovery(
    agent_home: Path,
    replacement: bytes,
) -> None:
    instruction = agent_home / "skills" / "stable" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: stable\ndescription: Stable body\n---\nbody\n")
    catalog = discover_skills(
        agent_home=AgentHome(agent_home),
        reserved_names=(),
        enable_always_load=False,
    )
    instruction.write_bytes(replacement)

    with pytest.raises(SkillUnavailableError) as failure:
        catalog.read_body(catalog.entries[0].metadata)

    assert failure.value.error.code == "skill_unavailable"
