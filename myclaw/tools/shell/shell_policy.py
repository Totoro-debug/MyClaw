"""Fixed first-version safe policy for Shell commands."""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from unicodedata import category

AUTOMATICALLY_ALLOWED_COMMANDS: Final = frozenset(
    {
        "pwd",
        "git status",
        "git status --short",
        "git diff --stat",
        "git diff --name-only",
    }
)
_GIT_FILTER_CONFIG_PATTERN: Final = r"^filter\..*\.(clean|smudge|process)$"


@dataclass(frozen=True, slots=True)
class _TrustedGitExecutable:
    path: Path
    identity: tuple[int, int, int, int]


def _capture_git_executable() -> _TrustedGitExecutable | None:
    discovered = shutil.which("git")
    if discovered is None:
        return None
    try:
        path = Path(discovered).resolve(strict=True)
        status = path.stat()
    except OSError:
        return None
    if not path.is_file() or path.suffix.casefold() != ".exe":
        return None
    return _TrustedGitExecutable(
        path=path,
        identity=(status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns),
    )


_TRUSTED_GIT_EXECUTABLE: Final = _capture_git_executable()


class ShellPolicyDenied(PermissionError):
    """Raised when a Shell request cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class ShellRequest:
    """One validated request passed to the operating-system Shell boundary."""

    command: str
    cwd: Path
    timeout: int
    workspace_root: Path | None = field(default=None, repr=False, compare=False)


def parse_shell_request(
    *,
    command: str,
    cwd: str,
    timeout: int,
    workspace: Path,
) -> ShellRequest:
    """Validate and normalize one Shell request within its Workspace cwd boundary."""
    if (
        not command.strip()
        or not cwd
        or not 60 <= timeout <= 600
        or any(category(character) == "Cc" for character in command)
    ):
        raise ShellPolicyDenied("invalid Shell command parameters")

    try:
        workspace_root = workspace.resolve(strict=True)
        target = Path(cwd)
        if not target.is_absolute():
            target = workspace_root / target
        resolved_cwd = target.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ShellPolicyDenied("Shell cwd cannot be resolved") from exc
    if (
        not workspace_root.is_dir()
        or not resolved_cwd.is_relative_to(workspace_root)
        or not resolved_cwd.is_dir()
    ):
        raise ShellPolicyDenied("Shell cwd must be a Workspace directory")
    return ShellRequest(
        command=command,
        cwd=resolved_cwd,
        timeout=timeout,
        workspace_root=workspace_root,
    )


def shell_command_is_allowed(
    command: str,
    *,
    cwd: Path | None = None,
    workspace: Path | None = None,
) -> bool:
    """Return whether one command matches the frozen safe read-only policy."""
    if command not in AUTOMATICALLY_ALLOWED_COMMANDS:
        return False
    if not command.startswith("git ") or cwd is None:
        return True
    git_executable = trusted_git_executable(workspace=workspace)
    if git_executable is None or _git_filters_may_run(cwd, git_executable):
        return False
    return True


def trusted_git_executable(*, workspace: Path | None = None) -> Path | None:
    captured = _TRUSTED_GIT_EXECUTABLE
    if captured is None:
        return None
    try:
        status = captured.path.stat()
        identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        workspace_root = workspace.resolve(strict=True) if workspace is not None else None
    except OSError:
        return None
    if identity != captured.identity:
        return None
    if workspace_root is not None and captured.path.is_relative_to(workspace_root):
        return None
    return captured.path


def _git_filters_may_run(cwd: Path, git_executable: Path) -> bool:
    try:
        completed = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "config",
                "--includes",
                "--null",
                "--name-only",
                "--get-regexp",
                _GIT_FILTER_CONFIG_PATTERN,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if completed.returncode == 1:
        return False
    if completed.returncode != 0:
        return True
    configured_filters = _configured_filter_names(completed.stdout)
    if configured_filters and "GIT_ATTR_SOURCE" in os.environ:
        return True
    selected_filters = _repository_attribute_filters(cwd, git_executable)
    return (
        configured_filters is None
        or selected_filters is None
        or not configured_filters.isdisjoint(selected_filters)
    )


def _configured_filter_names(output: bytes) -> frozenset[str] | None:
    names: set[str] = set()
    suffixes = (".clean", ".smudge", ".process")
    for raw_key in output.split(b"\0"):
        if not raw_key:
            continue
        key = os.fsdecode(raw_key)
        suffix = next((item for item in suffixes if key.endswith(item)), None)
        if not key.startswith("filter.") or suffix is None:
            return None
        name = key[len("filter.") : -len(suffix)]
        if not name:
            return None
        names.add(name)
    return frozenset(names)


def _repository_attribute_filters(
    cwd: Path,
    git_executable: Path,
) -> frozenset[str] | None:
    try:
        root_result = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "rev-parse",
                "--show-toplevel",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if root_result.returncode != 0:
        return frozenset()

    try:
        root = Path(os.fsdecode(root_result.stdout.strip())).resolve(strict=True)
    except (OSError, ValueError):
        return None

    try:
        global_attributes_result = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "config",
                "--includes",
                "--null",
                "--path",
                "--get-all",
                "core.attributesFile",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if global_attributes_result.returncode not in (0, 1):
        return None

    selected_filters: set[str] = set()
    for raw_path in global_attributes_result.stdout.split(b"\0"):
        if not raw_path:
            continue
        attributes_path = Path(os.fsdecode(raw_path))
        if not attributes_path.is_absolute():
            return None
        if attributes_path.exists():
            global_filters = _attribute_file_filters(attributes_path)
            if global_filters is None:
                return None
            selected_filters.update(global_filters)

    try:
        info_result = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "info/attributes",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if info_result.returncode != 0 or not info_result.stdout.strip():
        return None
    info_attributes = Path(os.fsdecode(info_result.stdout.strip()))
    if info_attributes.exists():
        info_filters = _attribute_file_filters(info_attributes)
        if info_filters is None:
            return None
        selected_filters.update(info_filters)

    walk_failed = False

    def record_walk_failure(_: OSError) -> None:
        nonlocal walk_failed
        walk_failed = True

    for directory, directories, files in os.walk(
        root,
        onerror=record_walk_failure,
        followlinks=False,
    ):
        directories[:] = [name for name in directories if name != ".git"]
        if ".gitattributes" not in files:
            continue
        file_filters = _attribute_file_filters(Path(directory) / ".gitattributes")
        if file_filters is None:
            return None
        selected_filters.update(file_filters)
    if walk_failed:
        return None
    return frozenset(selected_filters)


def _attribute_file_filters(path: Path) -> frozenset[str] | None:
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return None
    selected_filters: set[str] = set()
    for line in lines:
        if line.lstrip().startswith(b"#"):
            continue
        attributes = line.split()[1:]
        for attribute in attributes:
            if attribute == b"filter":
                selected_filters.add("true")
            elif attribute.startswith(b"filter="):
                raw_name = attribute[len(b"filter=") :]
                if not raw_name or b'"' in raw_name:
                    return None
                selected_filters.add(os.fsdecode(raw_name))
    return frozenset(selected_filters)
