import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml  # type: ignore[import-untyped]

from myclaw.management.commands import MANAGEMENT_COMMANDS

ROOT = Path(__file__).resolve().parents[1]

_ISSUE_202_PERSISTENCE_EVIDENCE = {
    "Session": (
        "tests/sessions/test_session.py::"
        "test_persist_writes_one_complete_compact_utf8_snapshot_atomically",
    ),
    "Summary": (
        "tests/memory/test_records.py::test_summary_entry_serializes_with_exactly_three_keys",
    ),
    "Cursor": (
        "tests/memory/test_memory_manager.py::"
        "test_manager_appends_and_claims_summaries_with_cursor_preadvance",
    ),
    "Long-term Memory": (
        "tests/memory/test_memory_manager.py::"
        "test_manager_reads_disk_and_refreshes_snapshot_after_an_edit",
    ),
    "Schedule": (
        "tests/scheduling/test_schedule_model.py::"
        "test_schedule_job_round_trips_the_strict_persisted_shape",
    ),
    "Artifact": (
        "tests/tools/test_base_tool.py::"
        "test_base_tool_result_handler_writes_a_bounded_workspace_artifact",
    ),
    "Dream System Job": (
        "tests/scheduling/test_schedule_dream.py::"
        "test_dream_registration_persists_a_hidden_recurring_system_job",
        "tests/scheduling/test_schedule_dream.py::"
        "test_exact_dream_registration_performs_zero_store_writes",
        "tests/scheduling/test_schedule_dream.py::"
        "test_due_dream_job_dispatches_directly_without_user_or_session_execution",
    ),
}
_ISSUE_202_ARCHITECTURE_EVIDENCE = (
    "tests/test_cli.py::test_cli_async_root_owns_lifetime_components_and_async_shutdown[normal]",
    "tests/agent/test_loop.py::"
    "test_agent_loop_constructs_each_generation_collaborator_once_without_side_effects",
    "tests/agent/test_message_bus.py::"
    "test_reset_clears_both_fifos_and_publishes_one_empty_snapshot",
    "tests/test_cli.py::test_cli_resume_publishes_current_only_after_target_activation",
    "tests/test_cli.py::test_legacy_runtime_module_is_not_discoverable",
)
_ISSUE_202_OWNER_NODES = (
    tuple(node for nodes in _ISSUE_202_PERSISTENCE_EVIDENCE.values() for node in nodes)
    + _ISSUE_202_ARCHITECTURE_EVIDENCE
)
_ISSUE_202_FORBIDDEN_RUNTIME_NAMES = (
    "RuntimeHost",
    "PreparedRuntime",
    "RuntimeBindings",
    "prepare_runtime",
    "_prepare_runtime",
    "MemoryTaskScheduler",
    "memory_scheduler",
    "RuntimeSkill" + "Snapshot",
    "build_runtime_skill" + "_snapshot",
    "SkillUnavailableError",
)
_ISSUE_202_FORBIDDEN_STRUCTURAL_NAMES = (
    *_ISSUE_202_FORBIDDEN_RUNTIME_NAMES,
    "Runtime",
    "Workspace",
    "read_body",
)
_ISSUE_202_FORBIDDEN_MODULES = (
    "myclaw.agent.runtime",
    "myclaw.agent.workspace",
    "myclaw.memory.memory_scheduler",
    "myclaw.agent.memory.memory_scheduler",
)
_ISSUE_202_FORBIDDEN_PARENT_IMPORTS = {
    "myclaw.agent": {"runtime", "workspace"},
    "myclaw.memory": {"memory_scheduler"},
    "myclaw.agent.memory": {"memory_scheduler"},
}

_STANDARDS_2_3_FORBIDDEN_MODULES = (
    "myclaw.agent.repl",
    "myclaw.memory.memory_task",
    "myclaw.agent.memory.memory_task",
    "myclaw.terminal.repl",
)
_STANDARDS_2_3_FORBIDDEN_NAMES = {
    "LongTermMemoryStore",
    "ManagementDispatcher",
    "MemoryEditFileTool",
    "MemoryReadFileTool",
    "MemoryStore",
    "MemoryTaskModelRouter",
    "MemoryTaskResult",
    "SummaryAppender",
    "SummaryCursorStore",
    "SummaryStore",
    "WorkspaceFileMemoryStore",
    "_abandon_unstarted",
    "_close_sessions",
    "_publish_unlocked",
    "_register_dream_job_sync",
    "_register_system_job_sync",
    "_unbind_management",
    "_wait_for_abort",
    "last_foreground_route_status",
    "run_repl",
    "run_terminal_conversation",
}


def _issue_202_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _issue_202_class(tree: ast.AST, name: str) -> ast.ClassDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _issue_202_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _issue_202_method_names(class_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _issue_202_direct_method(
    class_node: ast.ClassDef,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _issue_202_parameter_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )


def _issue_202_attribute_call_lines(
    tree: ast.AST,
    owner: str,
    attribute: str,
) -> tuple[int, ...]:
    return tuple(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


def _issue_202_named_call_lines(tree: ast.AST, names: set[str]) -> tuple[int, ...]:
    return tuple(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
    )


def _issue_202_assignment_lines(
    tree: ast.AST,
    target: str,
    value: str | None,
) -> tuple[int, ...]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(candidate, ast.Name) and candidate.id == target for candidate in node.targets
        ):
            continue
        if value is None and isinstance(node.value, ast.Constant) and node.value.value is None:
            lines.append(node.lineno)
        if value is not None and isinstance(node.value, ast.Name) and node.value.id == value:
            lines.append(node.lineno)
    return tuple(lines)


def _issue_202_pytest_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTEST_"):
            environment.pop(name, None)
    for name in (
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    ):
        environment.pop(name, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def _issue_202_run_pytest(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_asyncio.plugin",
            *arguments,
        ],
        cwd=ROOT,
        env=_issue_202_pytest_environment(),
        capture_output=True,
        check=False,
        text=True,
    )


def _issue_202_diagnostic(
    label: str,
    result: subprocess.CompletedProcess[str],
) -> str:
    return (
        f"{label} failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _issue_202_normalize_node_id(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value).strip().replace("\\", "/")


def _issue_202_collected_node_ids(output: str, expected: tuple[str, ...]) -> tuple[str, ...]:
    expected_set = set(expected)
    return tuple(
        node
        for node in (_issue_202_normalize_node_id(line) for line in output.splitlines())
        if node in expected_set
    )


def _issue_202_junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    cases = tuple(root.iter("testcase"))
    failures = sum(
        case.find("failure") is not None or case.find("error") is not None for case in cases
    )
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - failures - skipped
    return len(cases), passed, failures, skipped


def _issue_202_stale_symbol_findings(tree: ast.AST, source: str) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _ISSUE_202_FORBIDDEN_STRUCTURAL_NAMES:
                findings.append(f"{source}:{node.lineno}: declaration {node.name}")
        if isinstance(node, ast.Name) and node.id in _ISSUE_202_FORBIDDEN_STRUCTURAL_NAMES:
            findings.append(f"{source}:{node.lineno}: name {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr in _ISSUE_202_FORBIDDEN_STRUCTURAL_NAMES:
                findings.append(f"{source}:{node.lineno}: attribute {node.attr}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _ISSUE_202_FORBIDDEN_MODULES or any(
                    alias.name.startswith(f"{module}.") for module in _ISSUE_202_FORBIDDEN_MODULES
                ):
                    findings.append(f"{source}:{node.lineno}: import {alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in _ISSUE_202_FORBIDDEN_MODULES or any(
                module.startswith(f"{forbidden}.") for forbidden in _ISSUE_202_FORBIDDEN_MODULES
            ):
                findings.append(f"{source}:{node.lineno}: import from {module}")
            if node.level and module in {"runtime", "workspace", "memory_scheduler"}:
                findings.append(f"{source}:{node.lineno}: relative import from {module}")
            for alias in node.names:
                if (
                    alias.name in _ISSUE_202_FORBIDDEN_STRUCTURAL_NAMES
                    or (node.level and alias.name in {"runtime", "workspace", "memory_scheduler"})
                    or alias.name in _ISSUE_202_FORBIDDEN_PARENT_IMPORTS.get(module, set())
                ):
                    findings.append(f"{source}:{node.lineno}: imported name {alias.name}")
    return findings


# The tracked corpus uses these simple inline/reference target forms. This is not a
# complete CommonMark parser, and deliberately does not claim to be one.
_INLINE_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]\n]*\]\((?P<target><[^>\n]+>|[^)\s\n]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\)"
)
_REFERENCE_MARKDOWN_LINK = re.compile(r"(?m)^\s*\[[^\]\n]+\]:\s*(?P<target><[^>\n]+>|\S+)")
_EXPECTED_REMOVED_MARKDOWN_PATHS = frozenset(
    {
        ROOT / "docs" / "cli-composition-root-implementation-plan.md",
        ROOT / "docs" / "issue-195-terminal-commit-cancellation-fix-plan.md",
        ROOT / "docs" / "issue-201-test-migration-ledger.md",
        ROOT / "docs" / "mcp-tool-support-implementation-plan.md",
        ROOT / "docs" / "mcp-tool-support-spec.md",
        ROOT / "docs" / "myclaw-personal-agent-prd.md",
        ROOT / "docs" / "release-readiness.md",
        ROOT / "docs" / "security-fault-review.md",
        ROOT / "docs" / "terminal-conversation-ui-design.md",
        ROOT / "myclaw" / "templates" / "blackboard.md",
        ROOT / "myclaw" / "templates" / "conversation-summary-system-prompt.md",
        ROOT / "myclaw" / "templates" / "conversation-summary-input.md",
        ROOT / "myclaw" / "templates" / "current-user-input.md",
        ROOT / "myclaw" / "templates" / "interrupted-assistant-content.md",
        ROOT / "myclaw" / "templates" / "memory-task-input.md",
        ROOT / "myclaw" / "templates" / "runtime-context.md",
        ROOT / "myclaw" / "templates" / "user-input.md",
    }
)


def _tracked_markdown_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ("git", "ls-files", "-z", "--", "*.md"),
        cwd=ROOT,
    ).decode("utf-8")
    return tuple(ROOT / Path(relative) for relative in output.split("\0") if relative)


def test_tracked_markdown_inventory_preserves_missing_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = b"README.md\0docs/unexpected-missing.md\0"
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: output)

    assert _tracked_markdown_paths() == (
        ROOT / "README.md",
        ROOT / "docs" / "unexpected-missing.md",
    )


def test_tracked_markdown_link_contract_rejects_unexpected_deleted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected = ROOT / "docs" / "unexpected-missing.md"
    monkeypatch.setitem(globals(), "_tracked_markdown_paths", lambda: (unexpected,))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *_args, **_kwargs: b"docs/unexpected-missing.md\n",
    )

    with pytest.raises(AssertionError, match=r"unexpected-missing\.md"):
        test_tracked_markdown_local_links_resolve()


def _markdown_link_targets(content: str) -> tuple[str, ...]:
    return tuple(
        match.group("target").removeprefix("<").removesuffix(">")
        for pattern in (_INLINE_MARKDOWN_LINK, _REFERENCE_MARKDOWN_LINK)
        for match in pattern.finditer(content)
    )


def _local_markdown_path(target: str) -> str | None:
    if target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    return path or None


def _adr_status(path: Path) -> object:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0] == "---", path
    try:
        closing = lines.index("---", 1)
    except ValueError as error:
        raise AssertionError(f"{path} has no closing frontmatter delimiter") from error
    frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    assert isinstance(frontmatter, dict), path
    return frontmatter.get("status")


def _adr_status_contract_issues(decisions: list[Path]) -> list[str]:
    numbers: dict[str, Path] = {}
    issues: list[str] = []
    for path in decisions:
        number = path.name.split("-", 1)[0]
        previous = numbers.get(number)
        if previous is not None:
            issues.append(f"duplicate ADR number {number}: {previous.name}, {path.name}")
        else:
            numbers[number] = path

    for path in decisions:
        number = path.name.split("-", 1)[0]
        status = _adr_status(path)
        if status == "accepted":
            continue
        if not isinstance(status, str):
            issues.append(f"{path.name}: missing ADR status")
            continue
        match = re.fullmatch(r"superseded by ADR-(?P<number>\d{4})", status)
        if match is None:
            issues.append(f"{path.name}: invalid ADR status {status!r}")
            continue
        superseder_number = match.group("number")
        if superseder_number == number:
            issues.append(f"{path.name}: ADR cannot supersede itself")
            continue
        if int(superseder_number) < int(number):
            issues.append(
                f"{path.name}: superseding ADR-{superseder_number} must have a later number"
            )
            continue
        superseder = numbers.get(superseder_number)
        if superseder is None:
            issues.append(f"{path.name}: superseding ADR-{superseder_number} does not exist")
            continue
        if _adr_status(superseder) != "accepted":
            issues.append(f"{path.name}: superseding ADR-{superseder_number} is not accepted")
    return issues


def test_distribution_declares_supported_loguru_release_range() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "loguru>=0.7.3,<0.8" in project["dependencies"]


def test_distribution_directly_declares_iana_timezone_database() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "tzdata>=2026.2" in project["dependencies"]


def test_distribution_directly_declares_host_timezone_discovery() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "tzlocal>=5,<6" in project["dependencies"]


def test_distribution_retires_prompt_toolkit_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any(
        dependency.casefold().startswith("prompt-toolkit") for dependency in project["dependencies"]
    )


def test_distribution_metadata_builds_one_host_neutral_wheel() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["myclaw"] == "myclaw.terminal.process_entry:run"
    assert "Operating System :: OS Independent" not in project["classifiers"]
    assert "Operating System :: Microsoft :: Windows" not in project["classifiers"]
    setup_path = ROOT / "setup.cfg"
    setup = setup_path.read_text(encoding="utf-8") if setup_path.exists() else ""
    assert "plat_name" not in setup


def _ignore_unclean_build_inputs(_directory: str, names: list[str]) -> set[str]:
    ignored = {".codegraph", ".git", ".pytest_cache", "build", "dist", "__pycache__"}
    return {name for name in names if name in ignored or name.endswith(".egg-info")}


def test_clean_distributions_omit_deleted_agent_module_and_import_cleanly(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    shutil.copytree(ROOT, source_root, ignore=_ignore_unclean_build_inputs)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    build_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(artifact_dir),
        ],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert build_result.returncode == 0, build_result.stderr

    sdists = tuple(artifact_dir.glob("myclaw-*.tar.gz"))
    wheels = tuple(artifact_dir.glob("myclaw-*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1

    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_members = {member.name.replace("\\", "/") for member in archive.getmembers()}
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_members = {member.replace("\\", "/") for member in archive.namelist()}

    deleted_modules = (
        "myclaw/agent/prompts.py",
        "myclaw/agent/runtime.py",
        "myclaw/agent/repl.py",
        "myclaw/memory/memory_task.py",
        "myclaw/agent/memory/memory_task.py",
        "myclaw/session/projection.py",
        "myclaw/agent/session/projection.py",
        "myclaw/terminal/repl.py",
    )
    for deleted_module in deleted_modules:
        assert not any(member.endswith(f"/{deleted_module}") for member in sdist_members)
        assert deleted_module not in wheel_members

    install_root = tmp_path / "clean-install"
    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    clean_import_dir = tmp_path / "clean-import"
    clean_import_dir.mkdir()
    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib\n"
                "import importlib.util\n"
                "import myclaw\n"
                "import myclaw.terminal.cli\n"
                "legacy_modules = (\n"
                "    'myclaw.agent." + "prompts',\n"
                "    'myclaw.agent.runtime',\n"
                "    'myclaw.agent.repl',\n"
                "    'myclaw.memory.memory_task',\n"
                "    'myclaw.agent.memory.memory_task',\n"
                "    'myclaw.session." + "projection',\n"
                "    'myclaw.agent.session." + "projection',\n"
                "    'myclaw.terminal.repl',\n"
                ")\n"
                "for legacy_module in legacy_modules:\n"
                "    try:\n"
                "        spec = importlib.util.find_spec(legacy_module)\n"
                "    except ModuleNotFoundError:\n"
                "        spec = None\n"
                "    assert spec is None\n"
                "    try:\n"
                "        importlib.import_module(legacy_module)\n"
                "    except ModuleNotFoundError:\n"
                "        pass\n"
                "    else:\n"
                "        raise AssertionError(f'deleted module is importable: {legacy_module}')\n"
            ),
        ],
        cwd=clean_import_dir,
        env={**os.environ, "PYTHONPATH": str(install_root)},
        capture_output=True,
        check=False,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr


def test_active_code_has_no_platform_support_gate() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "myclaw").rglob("*.py"))
    )
    for residue in ("UnsupportedPlatformError", "unsupported_platform", "SUPPORTED_PLATFORM_TAG"):
        assert residue not in production
    assert not (ROOT / "myclaw" / "platform_support.py").exists()
    assert not (ROOT / "myclaw" / "terminal" / "entrypoint.py").exists()


def test_obsolete_runtime_log_contract_surface_is_absent() -> None:
    obsolete_paths = (
        ROOT / "myclaw" / "runtime_log.py",
        ROOT / "myclaw" / "runtime_log_lock.py",
        ROOT / "myclaw" / "logging" / "diagnostics.py",
        ROOT / "tests" / "runtime_log",
        ROOT / "tests" / "fixtures" / "log_capture.py",
    )

    assert not [path for path in obsolete_paths if path.exists()]


def test_session_log_adr_publishes_the_risk_contract() -> None:
    required_contract = (
        "same-session concurrency is unsupported",
        "unbounded queue",
        "infinite drain",
        "no per-record fsync",
        "no active redaction",
        "no control escaping",
        "per-session retention",
        "legacy agent home runtime log files remain untouched",
    )

    path = ROOT / "docs" / "adr" / "0008-use-workspace-session-log.md"
    content = path.read_text(encoding="utf-8").lower()
    assert all(statement in content for statement in required_contract), path


def test_active_contract_docs_do_not_claim_the_removed_runtime_log_implementation() -> None:
    active_contracts = (
        ROOT / "CONTEXT.md",
        ROOT / "docs" / "adr" / "0007-use-host-adapters.md",
    )
    obsolete_claims = (
        "shared runtime log",
        "runtime log lock",
        "runtime log locking",
        "runtime log |",
        "首版无持久化 runtime log",
    )

    for path in active_contracts:
        content = path.read_text(encoding="utf-8").lower()
        assert not [claim for claim in obsolete_claims if claim in content], path


def test_application_modules_do_not_depend_on_standard_library_logging() -> None:
    violations: list[str] = []

    for path in sorted((ROOT / "myclaw").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "logging" for alias in node.names
            ):
                violations.append(f"{path}: imports logging")
            if isinstance(node, ast.ImportFrom) and node.module == "logging":
                violations.append(f"{path}: imports from logging")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"debug", "info", "warning", "error", "critical"}
                and len(node.args) > 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "%" in node.args[0].value
            ):
                violations.append(f"{path}:{node.lineno}: percent-style logging arguments")
        if "InterceptHandler" in source:
            violations.append(f"{path}: logging interception bridge")

    assert violations == []


def test_active_support_contract_matches_host_neutral_release_evidence() -> None:
    decision_path = ROOT / "docs" / "adr" / "0007-use-host-adapters.md"
    assert decision_path.exists()
    decision = decision_path.read_text(encoding="utf-8").lower()
    assert "status: accepted" in decision
    assert "filesystem" in decision
    assert "process tree" not in decision
    assert "owned-process" not in decision
    assert "runtime log locking" not in decision

    for claim in (
        "py3-none-any",
        "windows x64",
        "currently validated",
        "macos intel",
        "apple silicon",
        "unverified",
        "no supported-platform gate",
    ):
        assert claim in decision


def test_superseded_design_documents_are_absent() -> None:
    superseded = (
        ROOT / "Procedure.md",
        ROOT / "docs" / "agent-runtime-message-bus-design.md",
        ROOT / "docs" / "contracts-modularization-execution-plan.md",
        ROOT / "docs" / "myclaw-implementation-plan.md",
        ROOT / "docs" / "research" / "agent-tool-calling-parameter-validation.md",
        ROOT / "docs" / "research" / "terminal-tui-library-selection.md",
        ROOT / "docs" / "adr" / "0003-shell-permission-is-not-os-sandbox.md",
        ROOT / "docs" / "adr" / "0004-use-two-slot-runtime-log.md",
        ROOT / "docs" / "adr" / "0006-support-windows-only.md",
        ROOT / "docs" / "adr" / "0011-use-terminal-conversation-as-the-interactive-cli.md",
        ROOT / "docs" / "adr" / "0012-use-textual-and-capability-gated-enhanced-keyboard-input.md",
        ROOT / "docs" / "adr" / "0013-emit-model-call-completion-for-run-projection.md",
    )

    assert not [path for path in superseded if path.exists()]


def test_current_adrs_have_unique_numbers_and_valid_status_contract() -> None:
    decisions = sorted((ROOT / "docs" / "adr").glob("*.md"))

    assert _adr_status_contract_issues(decisions) == []


@pytest.mark.parametrize(
    ("documents", "expected"),
    [
        (
            {"0001-first.md": "accepted", "0001-second.md": "accepted"},
            "duplicate ADR number",
        ),
        ({"0001-first.md": None}, "missing ADR status"),
        ({"0001-first.md": "draft"}, "invalid ADR status"),
        ({"0001-first.md": "superseded by ADR-0001"}, "ADR cannot supersede itself"),
        (
            {
                "0001-first.md": "accepted",
                "0002-second.md": "superseded by ADR-0001",
            },
            "must have a later number",
        ),
        (
            {
                "0001-first.md": "Superseded by ADR-0002",
                "0002-second.md": "accepted",
            },
            "invalid ADR status",
        ),
        (
            {"0001-first.md": "superseded by ADR-0099"},
            "superseding ADR-0099 does not exist",
        ),
        (
            {
                "0001-first.md": "superseded by ADR-0002",
                "0002-second.md": "superseded by ADR-0003",
            },
            "superseding ADR-0002 is not accepted",
        ),
    ],
)
def test_adr_status_contract_rejects_invalid_relationships(
    tmp_path: Path,
    documents: dict[str, str | None],
    expected: str,
) -> None:
    paths: list[Path] = []
    for filename, status in documents.items():
        status_line = "title: fixture\n" if status is None else f"status: {status}\n"
        path = tmp_path / filename
        path.write_text(f"---\n{status_line}---\n", encoding="utf-8")
        paths.append(path)

    issues = _adr_status_contract_issues(sorted(paths))

    assert expected in "\n".join(issues)


def test_mcp_transport_evidence_uses_local_fixtures() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    mcp_tests = (ROOT / "tests" / "tools" / "test_mcp.py").read_text(encoding="utf-8")
    cli_mcp_tests = (ROOT / "tests" / "test_cli_mcp_lifecycle.py").read_text(encoding="utf-8")

    assert "mcp>=2,<3" in project["dependencies"]
    test_urls = re.findall(r"https?://[^\"']+", mcp_tests)
    assert test_urls
    assert all(urlsplit(url).hostname in {"127.0.0.1", "localhost"} for url in test_urls)
    for test_name in (
        "test_stdio_transport_connects_to_a_local_real_mcp_server",
        "test_streamable_http_transport_connects_to_a_local_real_mcp_server",
    ):
        assert test_name in mcp_tests

    full_flow_test = "test_cli_real_mcp_flow_persists_result_reuses_connection_and_closes"
    assert full_flow_test in cli_mcp_tests


def test_mcp_release_contract_excludes_out_of_scope_runtime_surfaces() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "myclaw").rglob("*.py"))
    )
    gateway = (ROOT / "myclaw" / "agent" / "tools" / "tool_gateway.py").read_text(encoding="utf-8")

    for removed_name in (
        "PreparedToolCall",
        "memory_context_too_large",
        "skill_context_too_large",
    ):
        assert removed_name not in production
    assert "self._schemas" not in gateway
    assert all("mcp" not in command.token.casefold() for command in MANAGEMENT_COMMANDS)


def test_tracked_markdown_local_links_resolve() -> None:
    tracked = _tracked_markdown_paths()

    missing: list[str] = []
    for source in tracked:
        if not source.exists():
            if source not in _EXPECTED_REMOVED_MARKDOWN_PATHS:
                missing.append(f"{source.relative_to(ROOT)}: tracked Markdown source is missing")
            continue
        content = source.read_text(encoding="utf-8")
        for target in _markdown_link_targets(content):
            local_path = _local_markdown_path(target)
            if local_path is None:
                continue
            candidate = (source.parent / local_path).resolve()
            if not candidate.exists():
                missing.append(f"{source.relative_to(ROOT)}: {target} -> {candidate}")

    assert missing == []


def test_issue_202_architecture_claims_match_source_ast_contracts() -> None:
    loaded_skill = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "skills" / "catalog.py"),
        "LoadedSkill",
    )
    assert {
        node.target.id
        for node in loaded_skill.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    } == {"metadata", "document", "always"}

    skill_loader = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "skills" / "catalog.py"),
        "SkillLoader",
    )
    assert _issue_202_method_names(skill_loader) == {
        "root",
        "skills",
        "metadata",
        "get",
        "resolve_manual",
        "load",
    }

    message_bus = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "agent" / "message_bus.py"),
        "MessageBus",
    )
    assert {
        node.name
        for node in message_bus.body
        if isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_")
    } == {
        "inbound_snapshot",
        "put_inbound",
        "get_inbound",
        "pause_inbound_delivery",
        "resume_inbound_delivery",
        "drain_inbound",
        "put_outbound",
        "get_outbound",
        "reset",
    }
    assert {
        node.name
        for node in message_bus.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    } == {"set_inbound_changed_callback", "unbind_inbound_changed_callback"}

    memory_manager = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "agent" / "memory" / "manager.py"),
        "MemoryManager",
    )
    assert _issue_202_method_names(memory_manager) == {
        "long_term_path",
        "append_summary",
        "claim_summaries",
        "read_long_term",
        "edit_long_term",
        "memory_snapshot",
    }

    dream = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "agent" / "memory" / "dream.py"),
        "Dream",
    )
    assert _issue_202_parameter_names(_issue_202_function(dream, "__init__")) == (
        "self",
        "memory_manager",
        "model_router",
        "batch_size",
        "memory_route_status",
    )
    assert _issue_202_method_names(dream) == {
        "run",
        "close",
        "wait_until_idle",
        "abort",
        "abort_and_wait",
    }

    schedule_tree = _issue_202_ast(ROOT / "myclaw" / "schedule" / "service.py")
    schedule_clock = _issue_202_class(schedule_tree, "ScheduleClock")
    assert _issue_202_method_names(schedule_clock) == {"now", "monotonic", "sleep"}
    schedule_service = _issue_202_class(schedule_tree, "ScheduleService")
    assert _issue_202_parameter_names(_issue_202_function(schedule_service, "__init__")) == (
        "self",
        "workspace_state",
        "clock",
        "execute_user_job",
        "execute_dream",
        "timezone_name",
    )
    assert _issue_202_parameter_names(
        _issue_202_function(schedule_service, "register_dream_job")
    ) == ("self", "schedule")


def test_issue_202_cli_source_records_cutover_and_shutdown_order() -> None:
    cli_tree = _issue_202_ast(ROOT / "myclaw" / "terminal" / "cli.py")
    replacement = _issue_202_function(cli_tree, "replace_agent_loop")
    preflight_lines = _issue_202_attribute_call_lines(replacement, "target", "preflight")
    quiesce_lines = _issue_202_attribute_call_lines(
        replacement, "terminal_app", "quiesce_for_rebind"
    )
    pause_lines = _issue_202_attribute_call_lines(
        replacement, "schedule_service", "pause_and_drain"
    )
    reset_lines = _issue_202_attribute_call_lines(replacement, "bus", "reset")
    rebind_lines = _issue_202_attribute_call_lines(replacement, "terminal_app", "rebind_agent_loop")
    start_lines = _issue_202_attribute_call_lines(replacement, "target", "start")
    resume_lines = _issue_202_attribute_call_lines(replacement, "schedule_service", "resume")
    current_none_lines = _issue_202_assignment_lines(replacement, "current_loop", None)
    current_target_lines = _issue_202_assignment_lines(replacement, "current_loop", "target")
    old_abort_lines = tuple(
        line
        for line in _issue_202_named_call_lines(replacement, {"abort_loop_once"})
        if current_none_lines and line > min(current_none_lines)
    )

    cutover = (
        min(quiesce_lines),
        min(pause_lines),
        min(line for line in current_none_lines if line > min(pause_lines)),
        min(old_abort_lines),
        min(reset_lines),
        min(rebind_lines),
        min(start_lines),
        min(line for line in current_target_lines if line > min(start_lines)),
        min(resume_lines),
    )
    assert min(preflight_lines) < cutover[0]
    assert cutover == tuple(sorted(cutover))

    conversation = _issue_202_function(cli_tree, "_run_cli_conversation")
    shutdown = next(
        node for node in conversation.body if isinstance(node, ast.Try) and node.finalbody
    )
    final_tree = ast.Module(body=shutdown.finalbody, type_ignores=[])
    close_lines = _issue_202_attribute_call_lines(final_tree, "active_loop", "close")
    assert len(close_lines) == 1
    assert _issue_202_attribute_call_lines(conversation, "active_loop", "close") == close_lines
    abort_lines = _issue_202_named_call_lines(final_tree, {"abort_loop_once"})
    assert abort_lines
    schedule_close_line = min(
        _issue_202_attribute_call_lines(final_tree, "schedule_service", "close")
    )
    mcp_close_line = min(_issue_202_attribute_call_lines(final_tree, "mcp_manager", "close"))
    assert all(schedule_close_line < line < mcp_close_line for line in (*close_lines, *abort_lines))
    shutdown_events = (
        min(_issue_202_attribute_call_lines(final_tree, "management", "deactivate")),
        min(_issue_202_attribute_call_lines(final_tree, "schedule_service", "pause_and_drain")),
        schedule_close_line,
        mcp_close_line,
        min(_issue_202_attribute_call_lines(final_tree, "dream", "close")),
        min(_issue_202_attribute_call_lines(final_tree, "router", "close")),
    )
    assert shutdown_events == tuple(sorted(shutdown_events))


def test_issue_202_release_closure_maps_real_persistence_and_architecture_nodes(
    tmp_path: Path,
) -> None:
    evidence_nodes = _ISSUE_202_OWNER_NODES

    assert len(evidence_nodes) == 14
    assert len(set(evidence_nodes)) == len(evidence_nodes)
    assert all(not node.startswith("tests/test_release_contract.py::") for node in evidence_nodes)
    assert all(
        (ROOT / node.partition("::")[0]).resolve() != Path(__file__).resolve()
        for node in evidence_nodes
    )

    collect = _issue_202_run_pytest("--collect-only", "-q", *evidence_nodes)
    assert collect.returncode == 0, _issue_202_diagnostic("mapped collect", collect)
    collected = _issue_202_collected_node_ids(collect.stdout, evidence_nodes)
    assert len(collected) == len(evidence_nodes), (
        f"expected {len(evidence_nodes)} mapped nodes in collection output, "
        f"found {len(collected)}: {collected}\n{collect.stdout}\n{collect.stderr}"
    )
    assert set(collected) == set(evidence_nodes)

    junit_path = tmp_path / "issue-202-owner-results.xml"
    execution = _issue_202_run_pytest(
        "-q",
        "--junitxml",
        str(junit_path),
        *evidence_nodes,
    )
    assert execution.returncode == 0, _issue_202_diagnostic("mapped execution", execution)
    assert junit_path.is_file(), "mapped execution did not produce its JUnit report"
    executed, passed, failures, skipped = _issue_202_junit_counts(junit_path)
    assert (executed, passed, failures, skipped) == (14, 14, 0, 0), (
        f"mapped JUnit counts were {(executed, passed, failures, skipped)}\n"
        f"stdout:\n{execution.stdout}\nstderr:\n{execution.stderr}"
    )


def test_issue_202_active_stale_symbol_scan_is_precise_and_empty() -> None:
    active_sources = [
        *sorted((ROOT / "myclaw").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
    ]
    violations: list[str] = []
    for path in active_sources:
        tree = _issue_202_ast(path)
        violations.extend(_issue_202_stale_symbol_findings(tree, path.relative_to(ROOT).as_posix()))

    allowed_fixture = ast.parse(
        "RuntimeStatus = object()\n"
        "legacy_runtime_name = 'RuntimeHost'\n"
        "assert legacy_runtime_name\n"
    )
    assert _issue_202_stale_symbol_findings(allowed_fixture, "allowed_fixture.py") == []
    stale_fixture = ast.parse(
        "class RuntimeHost: pass\n"
        "class Runtime: pass\n"
        "class Workspace: pass\n"
        "def read_body(): pass\n"
        "import myclaw.agent.runtime as legacy_runtime\n"
        "from myclaw.agent import workspace\n"
        "from myclaw.agent.memory.memory_scheduler import MemoryTaskScheduler\n"
    )
    stale_fixture_findings = _issue_202_stale_symbol_findings(
        stale_fixture,
        "stale_fixture.py",
    )
    assert len(stale_fixture_findings) == 8

    assert violations == []
    assert not (ROOT / "myclaw" / "agent" / "runtime.py").exists()
    assert not (ROOT / "myclaw" / "agent" / "workspace.py").exists()
    legacy_scheduler_module = "_".join(("memory", "scheduler"))
    assert not (ROOT / "myclaw" / "memory" / f"{legacy_scheduler_module}.py").exists()
    assert not (ROOT / "myclaw" / "agent" / "memory" / f"{legacy_scheduler_module}.py").exists()


def test_standards_2_3_legacy_interfaces_are_absent_from_source() -> None:
    violations: list[str] = []
    source_paths = (
        *sorted((ROOT / "myclaw").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
    )
    for path in source_paths:
        relative = path.relative_to(ROOT).as_posix()
        tree = _issue_202_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _STANDARDS_2_3_FORBIDDEN_NAMES:
                    violations.append(f"{relative}:{node.lineno}: declaration {node.name}")
            if isinstance(node, ast.Name) and node.id in _STANDARDS_2_3_FORBIDDEN_NAMES:
                violations.append(f"{relative}:{node.lineno}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in _STANDARDS_2_3_FORBIDDEN_NAMES:
                violations.append(f"{relative}:{node.lineno}: attribute {node.attr}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _STANDARDS_2_3_FORBIDDEN_MODULES:
                        violations.append(f"{relative}:{node.lineno}: import {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _STANDARDS_2_3_FORBIDDEN_MODULES:
                    violations.append(f"{relative}:{node.lineno}: import from {module}")

    for module in _STANDARDS_2_3_FORBIDDEN_MODULES:
        path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
        if path.exists():
            violations.append(f"{path.relative_to(ROOT).as_posix()}: deleted module exists")
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            violations.append(f"{module}: deleted module is discoverable")

    assert violations == []


def test_ticket_10_transition_only_agent_run_scaffolding_is_absent() -> None:
    production_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "myclaw").rglob("*.py"))
    )

    assert "_LegacyConversationCompactor" not in production_text
    assert "CompactionModelRouter" not in production_text
    assert "compaction_message_threshold" not in production_text
    assert "estimated_input_tokens" not in production_text
    assert "context_used_percent" not in production_text
    assert "AgentRunnerMemoryRouter" not in production_text
    assert "AgentRunnerModelRoute" not in production_text
    assert "IdentityAgentRunRequestPreparer" not in production_text

    management = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "management" / "service.py"),
        "ManagementViewService",
    )
    management_init = _issue_202_direct_method(management, "__init__")
    assert _issue_202_parameter_names(management_init) == (
        "self",
        "agent_home",
        "current_agent_loop",
        "workspace_state",
        "replace_agent_loop",
        "prepare_session_resume",
        "memory_manager",
        "dream",
        "schedule_status",
        "now",
        "monotonic",
        "reasoning_effort_control",
    )
    assert management_init.args.defaults == []
    assert all(default is None for default in management_init.args.kw_defaults)

    terminal = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "terminal" / "conversation.py"),
        "TerminalConversationApp",
    )
    terminal_init = _issue_202_direct_method(terminal, "__init__")
    assert _issue_202_parameter_names(terminal_init) == (
        "self",
        "bus",
        "control",
        "management_dispatcher",
        "monotonic",
        "skill_metadata",
    )
    management_index = tuple(argument.arg for argument in terminal_init.args.kwonlyargs).index(
        "management_dispatcher"
    )
    assert terminal_init.args.kw_defaults[management_index] is None

    agent_loop = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "agent" / "loop.py"),
        "AgentLoop",
    )
    assert "bus" not in _issue_202_method_names(agent_loop)

    summary_store = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "agent" / "memory" / "store.py"),
        "WorkspaceJsonlSummaryStore",
    )
    assert "append_summary" not in _issue_202_method_names(summary_store)

    schedule_store = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "schedule" / "store.py"),
        "WorkspaceScheduleStore",
    )
    assert _issue_202_parameter_names(_issue_202_direct_method(schedule_store, "_publish")) == (
        "self",
        "candidate",
    )


def test_issue_234_session_uses_only_the_terminal_agent_run_commit() -> None:
    session = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "agent" / "session" / "session.py"),
        "Session",
    )
    methods = _issue_202_method_names(session)

    assert "commit_agent_run" in methods
    assert "append_messages" not in methods

    production_findings: list[str] = []
    for path in sorted((ROOT / "myclaw").rglob("*.py")):
        tree = _issue_202_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == (
                "append_messages"
            ):
                production_findings.append(f"{path}:{node.lineno}: declaration")
            if isinstance(node, ast.Name) and node.id == "append_messages":
                production_findings.append(f"{path}:{node.lineno}: name")
            if isinstance(node, ast.Attribute) and node.attr == "append_messages":
                production_findings.append(f"{path}:{node.lineno}: attribute")

    test_findings: list[str] = []
    release_contract_path = ROOT / "tests" / "test_release_contract.py"
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if path == release_contract_path:
            continue
        tree = _issue_202_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == (
                "append_messages"
            ):
                test_findings.append(f"{path}:{node.lineno}: declaration")
            if isinstance(node, ast.Name) and node.id == "append_messages":
                test_findings.append(f"{path}:{node.lineno}: name")
            if isinstance(node, ast.Attribute) and node.attr == "append_messages":
                test_findings.append(f"{path}:{node.lineno}: attribute")

    documentation_findings = [
        path
        for path in _tracked_markdown_paths()
        if "append_messages" in path.read_text(encoding="utf-8")
    ]
    assert production_findings == []
    assert test_findings == []
    assert documentation_findings == []


def test_issue_234_status_uses_one_canonical_anchor_and_no_sticky_route_state() -> None:
    loop_tree = _issue_202_ast(ROOT / "myclaw" / "agent" / "loop.py")
    compactor_tree = _issue_202_ast(
        ROOT / "myclaw" / "agent" / "memory" / "conversation_compactor.py"
    )
    forbidden_names = {
        "_configured_chat_context_window",
        "_configured_chat_model",
        "_last_foreground_route_status",
        "_latest_main_agent_context",
        "_latest_main_agent_provenance",
        "_latest_main_agent_usage",
        "_main_agent_usage_history",
        "_remember_foreground_route_status",
        "configured_chat_context_window",
        "configured_chat_model",
    }
    findings: list[str] = []
    for path, tree in (
        (ROOT / "myclaw" / "agent" / "loop.py", loop_tree),
        (
            ROOT / "myclaw" / "agent" / "memory" / "conversation_compactor.py",
            compactor_tree,
        ),
    ):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in forbidden_names:
                    findings.append(f"{path}:{node.lineno}: declaration {node.name}")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                findings.append(f"{path}:{node.lineno}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                findings.append(f"{path}:{node.lineno}: attribute {node.attr}")
    assert findings == []

    anchor_definitions = [
        node
        for node in compactor_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "latest_main_agent_usage_anchor"
    ]
    assert len(anchor_definitions) == 1

    controller = _issue_202_class(compactor_tree, "AgentRunContextController")
    controller_init = _issue_202_direct_method(controller, "__init__")
    controller_anchor_calls = [
        node
        for node in ast.walk(controller_init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "latest_main_agent_usage_anchor"
    ]
    assert len(controller_anchor_calls) == 1

    loop = _issue_202_class(loop_tree, "AgentLoop")
    runtime_status = _issue_202_direct_method(loop, "runtime_status_input")
    status_anchor_calls = [
        node
        for node in ast.walk(runtime_status)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "latest_main_agent_usage_anchor"
    ]
    configured_route_calls = [
        node
        for node in ast.walk(runtime_status)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_configured_model_route_status"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "chat"
    ]
    assert len(status_anchor_calls) == 1
    assert len(configured_route_calls) == 1


def test_issue_234_controller_migration_scaffolding_is_absent() -> None:
    compactor_tree = _issue_202_ast(
        ROOT / "myclaw" / "agent" / "memory" / "conversation_compactor.py"
    )
    class_names = {node.name for node in compactor_tree.body if isinstance(node, ast.ClassDef)}
    removed_dtos = {"AgentRunContextPreparation", "AgentRunStagedValues"}

    assert removed_dtos.isdisjoint(class_names)
    assert {"AgentRunContextRequestPreparer", "AgentRunTerminalCommitValues"} <= class_names

    exports_assignment = next(
        node
        for node in compactor_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    exports = ast.literal_eval(exports_assignment.value)
    assert removed_dtos.isdisjoint(exports)

    controller = _issue_202_class(compactor_tree, "AgentRunContextController")
    controller_methods = _issue_202_method_names(controller)
    assert "as_request_preparer" not in controller_methods
    assert {
        "base_context_revision",
        "checked_context_revision",
        "current_user_compacted",
        "latest_usage_context",
        "pending_action_summary",
        "pending_compaction_usage",
        "pending_last_compacted",
        "staged_values",
    }.isdisjoint(controller_methods)
    assert "terminal_commit_values" in controller_methods
    assert {
        "_base_context_revision",
        "_checked_context_revision",
        "_latest_usage_context",
    }.isdisjoint(node.attr for node in ast.walk(controller) if isinstance(node, ast.Attribute))

    loop_tree = _issue_202_ast(ROOT / "myclaw" / "agent" / "loop.py")
    agent_loop = _issue_202_class(loop_tree, "AgentLoop")
    new_agent_run_context = _issue_202_direct_method(agent_loop, "_new_agent_run_context")
    assert (
        len(
            _issue_202_named_call_lines(
                new_agent_run_context,
                {"AgentRunContextRequestPreparer"},
            )
        )
        == 1
    )

    production_constructor_calls: list[tuple[str, int]] = []
    removed_preparer_findings: list[tuple[str, int, str]] = []
    for path in sorted((ROOT / "myclaw").rglob("*.py")):
        tree = _issue_202_ast(path)
        relative = path.relative_to(ROOT).as_posix()
        production_constructor_calls.extend(
            (relative, line)
            for line in _issue_202_named_call_lines(
                tree,
                {"AgentRunContextRequestPreparer"},
            )
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "as_request_preparer"
            ):
                removed_preparer_findings.append((relative, node.lineno, "definition"))
            if isinstance(node, ast.Name) and node.id == "as_request_preparer":
                removed_preparer_findings.append((relative, node.lineno, "name"))
            if isinstance(node, ast.Attribute) and node.attr == "as_request_preparer":
                removed_preparer_findings.append((relative, node.lineno, "attribute"))

    assert len(production_constructor_calls) == 1
    assert removed_preparer_findings == []

    run_context = _issue_202_class(
        loop_tree,
        "_AgentRunContext",
    )
    run_context_fields = {
        node.target.id
        for node in run_context.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "request_preparer" not in run_context_fields
