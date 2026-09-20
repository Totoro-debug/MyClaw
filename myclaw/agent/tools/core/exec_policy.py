"""Host-neutral Exec facts and conservative shared risk matching."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Final, Literal

type ExecShellFamily = Literal["bash", "powershell", "pwsh"]
type ExecShellSelector = Literal["auto", "powershell", "pwsh"]
type ExecPlatform = Literal["posix", "windows"]
type ExecSyntaxConfidence = Literal["high", "medium", "low", "unknown"]
type ExecInspectorStatus = Literal["available", "uncertain", "failed", "timeout"]
type ExecPathRole = Literal["read", "write"]
type ExecIdentityKind = Literal[
    "builtin",
    "cmdlet",
    "native",
    "alias",
    "function",
    "script",
    "shim",
    "workspace",
    "ambiguous",
    "unknown",
]

EXEC_CONFIRMATION_REASON: Final = (
    "Exec inspection was unavailable or uncertain and requires confirmation."
)
EXEC_CATASTROPHIC_REASON: Final = (
    "The Exec command matches a known catastrophic operation and requires confirmation."
)


@dataclass(frozen=True, slots=True)
class ResolvedExecShell:
    """The immutable process-lifetime shell selection shared by inspection and execution."""

    selector: ExecShellSelector
    platform: ExecPlatform
    family: ExecShellFamily
    executable: str | None
    flags: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    available: bool
    diagnostic: str | None = None
    version: tuple[int, ...] | None = None

    @property
    def canonical_executable(self) -> str | None:
        """Return the selected executable identity without exposing PATH contents."""
        return self.executable

    @property
    def env(self) -> dict[str, str]:
        """Return a detached environment mapping for one host process."""
        return dict(self.environment)

    @property
    def shell(self) -> ExecShellFamily:
        """Return the canonical host family."""
        return self.family


@dataclass(frozen=True, slots=True)
class ExecCommandIdentity:
    """A command identity observed by a host inspector."""

    requested: str
    resolved: str | None = None
    kind: ExecIdentityKind = "unknown"


@dataclass(frozen=True, slots=True)
class ExecPathAccess:
    """A path operand and its coarse read/write role."""

    path: str
    role: ExecPathRole


@dataclass(frozen=True, slots=True)
class ExecDynamicConstruct:
    """A syntax construct whose target or effects are not statically fixed."""

    kind: str
    expression: str


@dataclass(frozen=True, slots=True)
class CatastrophicMatch:
    """One fixed matcher hit; policy mapping belongs outside the Host."""

    rule: str
    evidence: str


@dataclass(frozen=True, slots=True)
class ExecAssessment:
    """Shared inspection vocabulary returned by every Exec Host."""

    syntax_confidence: ExecSyntaxConfidence
    syntax_uncertain: bool
    command_identities: tuple[ExecCommandIdentity, ...] = ()
    file_accesses: tuple[ExecPathAccess, ...] = ()
    network_targets: tuple[str, ...] = ()
    dynamic_constructs: tuple[ExecDynamicConstruct, ...] = ()
    catastrophic_matches: tuple[CatastrophicMatch, ...] = ()
    inspector_status: ExecInspectorStatus = "available"
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_identities", tuple(self.command_identities))
        object.__setattr__(self, "file_accesses", tuple(self.file_accesses))
        object.__setattr__(self, "network_targets", tuple(self.network_targets))
        object.__setattr__(self, "dynamic_constructs", tuple(self.dynamic_constructs))
        object.__setattr__(self, "catastrophic_matches", tuple(self.catastrophic_matches))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def uncertain(self) -> bool:
        """Return whether inspection failed closed to an unknown classification."""
        return self.syntax_uncertain or self.inspector_status != "available"

    @property
    def confirmation_reason(self) -> str | None:
        """Return a legacy-compatible reason while later policy consumes typed facts."""
        if self.catastrophic_matches:
            return EXEC_CATASTROPHIC_REASON
        if self.uncertain:
            return EXEC_CONFIRMATION_REASON
        return None

    @classmethod
    def uncertain_result(
        cls,
        reason: str,
        *,
        status: ExecInspectorStatus = "uncertain",
        catastrophic: tuple[CatastrophicMatch, ...] = (),
    ) -> ExecAssessment:
        return cls(
            syntax_confidence="unknown",
            syntax_uncertain=True,
            catastrophic_matches=catastrophic,
            inspector_status=status,
            diagnostics=(reason,),
        )


@dataclass(frozen=True, slots=True)
class ExecOutcome:
    """Raw host process output. It intentionally has no Tool-layer fields."""

    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False

    @property
    def returncode(self) -> int | None:
        """Expose subprocess terminology without changing the raw boundary."""
        return self.exit_code


_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s\"'`<>]+",
    re.IGNORECASE,
)
_ASSIGNMENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[;|&\n])\s*[A-Za-z_][A-Za-z0-9_]*\s*=",
)
_SUBSTITUTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$\(|`[^`]*`|<\([^)]*\)", re.DOTALL)
_VARIABLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})")
_REDIRECTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:^|\s)(?:\d*>>?|\d*<)\s*[^\s]+")
_POWERSHELL_SCRIPTBLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{.*\}",
    re.DOTALL,
)


def catastrophic_matches(command: str) -> tuple[CatastrophicMatch, ...]:
    """Return invocation-aware catastrophic facts without deciding a permission level."""
    if not isinstance(command, str):
        raise TypeError("Exec command must be a string")
    matches: list[CatastrophicMatch] = []

    def add(rule: str, evidence: str) -> None:
        if all(existing.rule != rule for existing in matches):
            matches.append(CatastrophicMatch(rule=rule, evidence=evidence))

    for segment in _command_segments(command):
        invocation = _unwrap_invocation(segment)
        if not invocation:
            continue
        name = _command_basename(invocation[0])
        evidence = " ".join(invocation)

        deletion = _deletion_facts(invocation)
        if deletion is not None:
            recursive, force, targets = deletion
            if recursive and force and any(_is_broad_target(target) for target in targets):
                add("broad-recursive-force-delete", evidence)
            if recursive and force and any(_is_dynamic_target(target) for target in targets):
                add("dynamic-recursive-force-delete", evidence)

        if name in {
            "diskpart",
            "format",
            "initialize-disk",
            "format-volume",
            "clear-disk",
            "wipefs",
        } or name.startswith("mkfs."):
            add("disk-initialization-or-format", evidence)
        if name in {"remove-volume", "remove-partition", "remove-disk"}:
            add("volume-or-partition-removal", evidence)
        if name == "delete" and len(invocation) > 1 and invocation[1].lower() in {
            "volume",
            "partition",
        }:
            add("volume-or-partition-removal", evidence)

        if _overwrites_device(invocation):
            add("device-overwrite", evidence)

        git = _git_invocation(invocation)
        if git is not None:
            subcommand, arguments = git
            if subcommand == "clean" and _git_clean_is_catastrophic(arguments):
                add("git-clean-all", evidence)
            elif subcommand == "reset" and any(
                argument.lower() == "--hard" for argument in arguments
            ):
                add("git-reset-hard", evidence)
            elif subcommand == "checkout" and _git_checkout_is_broad_force(arguments):
                add("git-force-checkout-or-restore", evidence)
            elif subcommand == "restore" and _git_restore_is_broad(arguments):
                add("git-force-checkout-or-restore", evidence)

        if name in {
            "shutdown",
            "reboot",
            "poweroff",
            "halt",
            "stop-computer",
            "restart-computer",
        }:
            add("shutdown-or-reboot", evidence)
        if (
            name in {"systemctl", "loginctl"}
            and len(invocation) > 1
            and invocation[1].lower() in {"reboot", "poweroff", "halt"}
        ):
            add("shutdown-or-reboot", evidence)

        nested = _nested_shell_command(invocation)
        if nested is not None:
            for nested_match in catastrophic_matches(nested):
                add(nested_match.rule, nested_match.evidence)

    unquoted = _unquoted_source(command)
    fork_bomb = re.search(
        r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:",
        unquoted,
    )
    if fork_bomb is not None:
        add("fork-bomb", fork_bomb.group(0))
    return tuple(matches)


def requires_legacy_destructive_confirmation(command: str) -> bool:
    """Preserve the pre-level Exec confirmation surface for destructive syntax."""
    if catastrophic_matches(command):
        return True
    for segment in _command_segments(command):
        invocation = _unwrap_invocation(segment)
        if not invocation:
            continue
        name = _command_basename(invocation[0])
        deletion = _deletion_facts(invocation)
        if deletion is not None:
            recursive, force, _targets = deletion
            if recursive or force:
                return True
        lowered = tuple(token.lower() for token in invocation[1:])
        if name in {"del", "erase"} and any(
            "f" in token[1:] for token in lowered if token.startswith("/")
        ):
            return True
        if name == "dd" and any(token.startswith("if=") for token in lowered):
            return True
    return False


def _command_segments(command: str) -> tuple[tuple[str, ...], ...]:
    segments: list[tuple[str, ...]] = []
    tokens: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0

    def flush_token() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    def flush_segment() -> None:
        flush_token()
        if tokens:
            segments.append(tuple(tokens))
            tokens.clear()

    while index < len(command):
        character = command[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character in {"`", "\\"} and index + 1 < len(command):
                next_character = command[index + 1]
                if character == "`" or next_character in {quote, "$", "`"}:
                    buffer.append(next_character)
                    index += 1
                else:
                    buffer.append(character)
            else:
                buffer.append(character)
        elif character in {"'", '"'}:
            quote = character
        elif character == "#" and not buffer:
            while index < len(command) and command[index] != "\n":
                index += 1
            flush_segment()
        elif character.isspace():
            flush_token()
            if character == "\n":
                flush_segment()
        elif character in ";|&":
            flush_segment()
        elif character in "<>":
            flush_token()
            operator = character
            if index + 1 < len(command) and command[index + 1] == character:
                operator += character
                index += 1
            tokens.append(operator)
        elif character in {"`", "\\"} and index + 1 < len(command):
            next_character = command[index + 1]
            if character == "`" or next_character.isspace() or next_character in {"'", '"'}:
                buffer.append(next_character)
                index += 1
            else:
                buffer.append(character)
        else:
            buffer.append(character)
        index += 1
    flush_segment()
    return tuple(segments)


def _unwrap_invocation(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = list(tokens)
    while values and values[0] in {"(", ")", "{", "}", "."}:
        values.pop(0)
    if values:
        values[0] = values[0].lstrip("({")
        if not values[0]:
            values.pop(0)
    while values and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", values[0]):
        values.pop(0)
    if not values:
        return ()
    name = _command_basename(values[0])
    if name == "sudo":
        values.pop(0)
        while values and values[0].startswith("-"):
            values.pop(0)
    elif name == "env":
        values.pop(0)
        while values and (
            values[0].startswith("-")
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", values[0]) is not None
        ):
            values.pop(0)
    elif name in {"command", "builtin", "nohup"}:
        values.pop(0)
        while values and values[0].startswith("-"):
            values.pop(0)
    return tuple(values)


def _command_basename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def _option_flags(arguments: tuple[str, ...]) -> tuple[set[str], set[str], tuple[str, ...]]:
    short: set[str] = set()
    long: set[str] = set()
    targets: list[str] = []
    options_done = False
    for argument in arguments:
        lowered = argument.lower()
        if not options_done and argument == "--":
            options_done = True
        elif not options_done and lowered.startswith("--"):
            long.add(lowered[2:].split("=", maxsplit=1)[0])
        elif not options_done and lowered.startswith("-") and len(lowered) > 1:
            short.update(lowered[1:])
        elif not options_done and lowered.startswith("/") and len(lowered) == 2:
            short.add(lowered[1])
        else:
            targets.append(argument)
    return short, long, tuple(targets)


def _deletion_facts(
    invocation: tuple[str, ...],
) -> tuple[bool, bool, tuple[str, ...]] | None:
    name = _command_basename(invocation[0])
    if name not in {"rm", "remove-item", "ri", "rd", "rmdir"}:
        return None
    short, long, targets = _option_flags(invocation[1:])
    recursive = bool({"r", "s"} & short or {"recursive", "recurse"} & long)
    force = bool({"f", "q"} & short or "force" in long)
    return recursive, force, targets


def _is_dynamic_target(target: str) -> bool:
    return any(marker in target for marker in ("$", "`", "*", "?", "$("))


def _is_broad_target(target: str) -> bool:
    lowered = target.strip().lower()
    if lowered in {
        "/",
        "\\",
        ".",
        "./",
        ".\\",
        "..",
        "../",
        "..\\",
        "~",
        "~/",
        "~\\",
    }:
        return True
    if lowered.rstrip("/\\") in {
        "$home",
        "${home}",
        "$env:userprofile",
        "%userprofile%",
    }:
        return True
    if re.fullmatch(r"[a-z]:[\\/]*", lowered) is not None:
        return True
    if re.fullmatch(r"/(?:home|users)/[^/]+[/]*", lowered) is not None:
        return True
    if re.fullmatch(r"[a-z]:[\\/]users[\\/][^\\/]+[\\/]*", lowered) is not None:
        return True
    return re.fullmatch(r"\\\\[^\\/]+[\\/][^\\/]+[\\/]*", target) is not None


def _is_device_path(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("of="):
        lowered = lowered[3:]
    return (
        re.fullmatch(r"/dev/(?:sd[a-z]\d*|nvme\w+|mmcblk\w+)", lowered) is not None
        or re.fullmatch(r"\\\\\.\\physicaldrive\d+", lowered) is not None
    )


def _overwrites_device(invocation: tuple[str, ...]) -> bool:
    name = _command_basename(invocation[0])
    arguments = invocation[1:]
    if name == "dd" and any(
        argument.lower().startswith("of=") and _is_device_path(argument)
        for argument in arguments
    ):
        return True
    if name in {"set-content", "out-file", "tee"} and any(
        _is_device_path(argument) for argument in arguments
    ):
        return True
    return any(
        argument in {">", ">>"}
        and index + 1 < len(arguments)
        and _is_device_path(arguments[index + 1])
        for index, argument in enumerate(arguments)
    )


def _git_invocation(invocation: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
    if _command_basename(invocation[0]) != "git":
        return None
    index = 1
    value_options = {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env"}
    while index < len(invocation):
        argument = invocation[index]
        if argument in value_options:
            index += 2
        elif argument.startswith("-C") and argument != "-C":
            index += 1
        elif argument.startswith("-"):
            index += 1
        else:
            return argument.lower(), invocation[index + 1 :]
    return None


def _git_clean_is_catastrophic(arguments: tuple[str, ...]) -> bool:
    short, long, _targets = _option_flags(arguments)
    force = "f" in short or "force" in long
    directories = "d" in short or "directories" in long
    return force and directories and "x" in short


def _git_checkout_is_broad_force(arguments: tuple[str, ...]) -> bool:
    short, long, targets = _option_flags(arguments)
    if "f" not in short and "force" not in long:
        return False
    if "--" not in arguments:
        return True
    return any(_is_broad_target(target) for target in targets)


def _git_restore_is_broad(arguments: tuple[str, ...]) -> bool:
    _short, _long, targets = _option_flags(arguments)
    return any(_is_broad_target(target) for target in targets)


def _nested_shell_command(invocation: tuple[str, ...]) -> str | None:
    name = _command_basename(invocation[0])
    selectors = {
        "bash": {"-c"},
        "sh": {"-c"},
        "pwsh": {"-command", "-c"},
        "powershell": {"-command", "-c"},
        "cmd": {"/c"},
    }
    accepted = selectors.get(name)
    if accepted is None:
        return None
    for index, argument in enumerate(invocation[1:], start=1):
        if argument.lower() in accepted and index + 1 < len(invocation):
            return invocation[index + 1]
    return None


def _unquoted_source(command: str) -> str:
    result: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        character = command[index]
        if quote is not None:
            if character == quote:
                quote = None
            result.append(" ")
        elif character in {"'", '"'}:
            quote = character
            result.append(" ")
        elif character == "#":
            while index < len(command) and command[index] != "\n":
                result.append(" ")
                index += 1
            result.append("\n")
        else:
            result.append(character)
        index += 1
    return "".join(result)


def assess_command(
    command: str,
    *,
    family: ExecShellFamily,
    syntax_confidence: ExecSyntaxConfidence,
    syntax_uncertain: bool,
    command_names: tuple[str, ...] = (),
    diagnostics: tuple[str, ...] = (),
    inspector_status: ExecInspectorStatus = "available",
) -> ExecAssessment:
    """Build a shared assessment from parser facts and conservative text roles."""
    names = command_names or _command_names(command)
    identities = tuple(
        ExecCommandIdentity(
            requested=name,
            kind=(
                "cmdlet"
                if family in {"powershell", "pwsh"} and "-" in name
                else "builtin"
                if name in _BUILTINS["powershell" if family in {"powershell", "pwsh"} else "bash"]
                else "unknown"
            ),
        )
        for name in names
    )
    dynamic = _dynamic_constructs(command, family=family)
    return ExecAssessment(
        syntax_confidence=syntax_confidence,
        syntax_uncertain=syntax_uncertain,
        command_identities=identities,
        file_accesses=_path_accesses(command, names, family=family),
        network_targets=tuple(match.group(0).rstrip(".,;:!?)]}") for match in _URL_PATTERN.finditer(command)),
        dynamic_constructs=dynamic,
        catastrophic_matches=catastrophic_matches(command),
        inspector_status=inspector_status,
        diagnostics=diagnostics,
    )


def _command_names(command: str) -> tuple[str, ...]:
    names: list[str] = []
    for segment in re.split(r"(?:^|[;|&\n])", command):
        candidate = segment.strip()
        candidate = re.sub(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+", "", candidate)
        candidate = re.sub(r"^\s*&\s*", "", candidate)
        match = re.match(r"([^\s;|&]+)", candidate)
        if match is not None:
            value = match.group(1).strip("'\"")
            if value and value not in names:
                names.append(value)
    return tuple(names)


def _dynamic_constructs(command: str, *, family: ExecShellFamily) -> tuple[ExecDynamicConstruct, ...]:
    constructs: list[ExecDynamicConstruct] = []
    if _ASSIGNMENT_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("assignment", "variable assignment"))
    if _SUBSTITUTION_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("substitution", "command or process substitution"))
    if _VARIABLE_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("variable", "variable expansion"))
    if _REDIRECTION_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("redirection", "stream redirection"))
    if (
        family in {"powershell", "pwsh"}
        and _POWERSHELL_SCRIPTBLOCK_PATTERN.search(command) is not None
    ):
        constructs.append(ExecDynamicConstruct("scriptblock", "PowerShell scriptblock"))
    if re.search(r"\b(?:eval|source|invoke-expression|start-job)\b", command, re.IGNORECASE):
        constructs.append(ExecDynamicConstruct("indirect-invocation", "indirect invocation"))
    return tuple(constructs)


def _path_accesses(
    command: str,
    names: tuple[str, ...],
    *,
    family: ExecShellFamily,
) -> tuple[ExecPathAccess, ...]:
    try:
        tokens = shlex.split(command, posix=family == "bash")
    except ValueError:
        tokens = re.findall(r"[^\s]+", command)
    writes = {
        "rm",
        "rmdir",
        "del",
        "erase",
        "touch",
        "mkdir",
        "mv",
        "cp",
        "set-content",
        "add-content",
        "clear-content",
        "new-item",
        "remove-item",
        "copy-item",
        "move-item",
    }
    first = names[0].lower() if names else ""
    role: ExecPathRole = "write" if first in writes else "read"
    accesses: list[ExecPathAccess] = []
    for token in tokens:
        clean = token.lstrip("0123456789").lstrip("><")
        if not _looks_like_path(clean):
            continue
        accesses.append(ExecPathAccess(path=clean, role=role))
    if re.search(r"(?:^|\s)\d*>>?\s*", command) is not None:
        for target in re.findall(r"(?:^|\s)\d*>>?\s*([^\s]+)", command):
            accesses.append(ExecPathAccess(path=target, role="write"))
    return tuple(dict.fromkeys(accesses))


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("/", "./", "../", "~", "$", "\\\\"))
        or "/" in value
        or "\\" in value
        or re.fullmatch(r"[A-Za-z]:", value) is not None
        or any(character in value for character in "*?")
    )


_BUILTINS: Final[dict[ExecShellFamily, frozenset[str]]] = {
    "bash": frozenset({"cd", "echo", "printf", "pwd", "read", "test", "true", "false"}),
    "powershell": frozenset({"echo", "cd", "pwd", "where", "write-output"}),
}


__all__ = [
    "CatastrophicMatch",
    "ExecAssessment",
    "ExecCommandIdentity",
    "ExecDynamicConstruct",
    "ExecIdentityKind",
    "ExecInspectorStatus",
    "ExecOutcome",
    "ExecPathAccess",
    "ExecPathRole",
    "ExecPlatform",
    "ExecShellFamily",
    "ExecShellSelector",
    "ExecSyntaxConfidence",
    "ResolvedExecShell",
    "assess_command",
    "catastrophic_matches",
    "requires_legacy_destructive_confirmation",
]
