"""Runtime-lifetime discovery of user-authored Skill metadata."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml  # type: ignore[import-untyped]
from loguru import logger

from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

_NAME_PATTERN = re.compile(r"[a-z_-][a-z0-9_-]{0,63}")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Validated metadata retained for one Skill candidate."""

    name: str
    description: str
    path: Path


@dataclass(frozen=True, slots=True)
class SkillEntry:
    """One immutable Skill Catalog entry."""

    metadata: SkillMetadata
    always_body: str | None


@dataclass(frozen=True, slots=True)
class ManualSkillInvocation:
    """One validated manual Skill invocation for a foreground Agent Run."""

    metadata: SkillMetadata
    request: str
    body: str


class SkillUnavailableError(Exception):
    """A complete Skill body could not be read from its catalog snapshot."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class SkillCatalog:
    """Ordered immutable Skill metadata and a name lookup snapshot."""

    def __init__(self, *, root: Path, entries: Iterable[SkillEntry]) -> None:
        self._root = Path(root)
        self._entries = tuple(entries)
        self._by_name: Mapping[str, SkillEntry] = MappingProxyType(
            {entry.metadata.name: entry for entry in self._entries}
        )

    @property
    def entries(self) -> tuple[SkillEntry, ...]:
        """Return the ordered immutable entry snapshot."""
        return self._entries

    @property
    def root(self) -> Path:
        """Return the canonical Skill root used by this snapshot."""
        return self._root

    def get(self, name: str) -> SkillEntry | None:
        """Return the entry for an exact Skill name, when present."""
        return self._by_name.get(name)

    def read_body(self, metadata: SkillMetadata) -> str:
        """Read and revalidate one catalog Skill body without retaining it."""
        return _read_complete_body(self, metadata, require_always=False)

    def resolve_manual(self, raw_input: str) -> ManualSkillInvocation | None:
        """Resolve one exact slash invocation and load its current complete body."""
        if not isinstance(raw_input, str):
            raise TypeError("Skill input must be a string")
        if not raw_input.startswith("/"):
            return None

        token = raw_input[1:]
        delimiter = next(
            (index for index, character in enumerate(token) if character.isspace()),
            None,
        )
        if delimiter is None:
            name = token
            request = ""
        else:
            name = token[:delimiter]
            request = token[delimiter + 1 :]
        entry = self.get(name)
        if entry is None:
            return None
        return ManualSkillInvocation(
            metadata=entry.metadata,
            request=request,
            body=self.read_body(entry.metadata),
        )


def discover_skills(
    *,
    agent_home: AgentHome,
    reserved_names: Collection[str],
    enable_always_load: bool,
) -> SkillCatalog:
    """Build one immutable Skill Catalog snapshot."""
    reserved = {name.removeprefix("/") for name in reserved_names}
    root = agent_home.skills_directory.resolve()
    if not root.is_dir():
        logger.info("Discovered Skills count=0")
        return SkillCatalog(root=root, entries=())

    entries: list[SkillEntry] = []
    always_names: set[str] = set()
    names: set[str] = set()
    try:
        children = tuple(root.iterdir())
    except OSError:
        logger.warning("Skipping Skill discovery path={} reason=Skill root is unavailable", root)
        logger.info("Discovered Skills count=0")
        return SkillCatalog(root=root, entries=())

    candidates: list[tuple[Path, Path]] = []
    for candidate in children:
        if not candidate.is_dir():
            continue
        try:
            canonical_candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            _log_invalid(candidate, "Skill directory canonical path is unavailable")
            continue
        candidates.append((canonical_candidate, candidate))

    for canonical_candidate, candidate in sorted(
        candidates,
        key=lambda item: (str(item[0]), str(item[1])),
    ):
        if not canonical_candidate.is_relative_to(root):
            _log_invalid(candidate, "Skill directory canonical path escapes Skill root")
            continue
        instruction = candidate / "SKILL.md"
        try:
            status = instruction.stat()
        except OSError:
            _log_invalid(candidate, "SKILL.md is unavailable")
            continue
        if not HOST_FILESYSTEM.is_regular_file(status):
            _log_invalid(candidate, "SKILL.md is not a regular file")
            continue
        try:
            path = instruction.resolve(strict=True)
        except (OSError, RuntimeError):
            _log_invalid(candidate, "SKILL.md canonical path is unavailable")
            continue
        if not path.is_relative_to(root):
            _log_invalid(candidate, "SKILL.md canonical path escapes Skill root")
            continue
        metadata, always_value, reason = _read_metadata(
            instruction,
            path,
            interpret_always=enable_always_load,
        )
        if metadata is None:
            _log_invalid(candidate, reason)
            continue
        if enable_always_load:
            if always_value is not _MISSING and not isinstance(always_value, bool):
                logger.warning(
                    "Ignoring non-boolean Skill always field path={} reason=always must be a boolean",
                    candidate,
                )
        if metadata.name in reserved:
            _log_invalid(candidate, "name is reserved")
            continue
        if metadata.name in names:
            _log_invalid(candidate, "name is duplicated")
            continue
        names.add(metadata.name)
        if enable_always_load and always_value is True:
            always_names.add(metadata.name)
        entries.append(SkillEntry(metadata=metadata, always_body=None))
    logger.info("Discovered Skills count={}", len(entries))
    catalog = SkillCatalog(root=root, entries=entries)
    if not always_names:
        return catalog

    frozen_entries = [
        SkillEntry(
            metadata=entry.metadata,
            always_body=(
                _read_complete_body(catalog, entry.metadata, require_always=True)
                if entry.metadata.name in always_names
                else None
            ),
        )
        for entry in catalog.entries
    ]
    return SkillCatalog(root=root, entries=frozen_entries)


def _read_metadata(
    instruction: Path,
    path: Path,
    *,
    interpret_always: bool,
) -> tuple[SkillMetadata | None, object, str]:
    try:
        with instruction.open("rb") as stream:
            opening = stream.readline()
            if not _is_delimiter(opening):
                return None, _MISSING, "frontmatter opening delimiter is missing"
            frontmatter: list[bytes] = []
            for line in stream:
                if _is_delimiter(line):
                    break
                frontmatter.append(line)
            else:
                return None, _MISSING, "frontmatter closing delimiter is missing"
    except OSError:
        return None, _MISSING, "SKILL.md could not be read"
    try:
        document: object = yaml.safe_load(b"".join(frontmatter).decode("utf-8"))
    except UnicodeDecodeError:
        return None, _MISSING, "frontmatter is not valid UTF-8"
    except yaml.YAMLError:
        return None, _MISSING, "frontmatter YAML is invalid"
    metadata, reason = _validate_metadata(document, path)
    if metadata is None:
        return None, _MISSING, reason
    always_value = _always_value(document) if interpret_always else _MISSING
    return metadata, always_value, ""


def _parse_document(
    content: str,
    path: Path,
    *,
    interpret_always: bool,
) -> tuple[SkillMetadata | None, object, str]:
    lines = content.splitlines(keepends=True)
    if not lines or not _is_text_delimiter(lines[0]):
        return None, _MISSING, content
    frontmatter: list[str] = []
    body_start = None
    for index, line in enumerate(lines[1:], start=1):
        if _is_text_delimiter(line):
            body_start = index + 1
            break
        frontmatter.append(line)
    if body_start is None:
        return None, _MISSING, content
    try:
        document: object = yaml.safe_load("".join(frontmatter))
    except yaml.YAMLError:
        return None, _MISSING, content
    metadata, _reason = _validate_metadata(document, path)
    always_value = _always_value(document) if interpret_always else _MISSING
    return metadata, always_value, "".join(lines[body_start:])


def _read_complete_body(
    catalog: SkillCatalog,
    metadata: SkillMetadata,
    *,
    require_always: bool,
) -> str:
    if not isinstance(metadata, SkillMetadata):
        raise _skill_unavailable()
    entry = catalog._by_name.get(metadata.name)
    if entry is None or entry.metadata != metadata:
        raise _skill_unavailable()

    try:
        io_path = HOST_FILESYSTEM.path_for_io(metadata.path)
        with io_path.open("rb") as stream:
            path = HOST_FILESYSTEM.require_opened_contained_regular_file(
                stream.fileno(),
                metadata.path,
                within=catalog.root,
            )
            if path != metadata.path:
                raise ValueError("Skill path no longer matches the catalog snapshot")
            raw_content = stream.read()
        if not path.is_relative_to(catalog.root):
            raise ValueError("Skill path is no longer contained by the catalog root")
    except (OSError, RuntimeError, ValueError) as error:
        raise _skill_unavailable() from error

    try:
        content = raw_content.decode("utf-8")
    except UnicodeError as error:
        raise _skill_unavailable() from error

    parsed, always_value, body = _parse_document(
        content,
        path,
        interpret_always=require_always,
    )
    if parsed is None or parsed != metadata or (require_always and always_value is not True):
        raise _skill_unavailable()
    return body


def _always_value(document: object) -> object:
    if not isinstance(document, Mapping) or "always" not in document:
        return _MISSING
    return document["always"]


def _validate_metadata(document: object, path: Path) -> tuple[SkillMetadata | None, str]:
    if not isinstance(document, Mapping):
        return None, "frontmatter root is not a mapping"
    name = document.get("name")
    description = document.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        return None, "name and description must be strings"
    normalized_name = name.strip()
    normalized_description = description.strip()
    if not _NAME_PATTERN.fullmatch(normalized_name) or not 1 <= len(normalized_description) <= 1024:
        return None, "name or description is outside the accepted bounds"
    return (
        SkillMetadata(
            name=normalized_name,
            description=normalized_description,
            path=path,
        ),
        "",
    )


def _is_delimiter(line: bytes) -> bool:
    return line.rstrip(b"\r\n") == b"---"


def _is_text_delimiter(line: str) -> bool:
    return line.rstrip("\r\n") == "---"


def _skill_unavailable() -> SkillUnavailableError:
    return SkillUnavailableError(ErrorInfo("skill_unavailable", "Skill body is unavailable."))


def _log_invalid(candidate: Path, reason: str) -> None:
    logger.warning("Skipping invalid Skill candidate path={} reason={}", candidate, reason)


__all__ = [
    "ManualSkillInvocation",
    "SkillCatalog",
    "SkillEntry",
    "SkillMetadata",
    "SkillUnavailableError",
    "discover_skills",
]
