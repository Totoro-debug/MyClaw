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

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]

_ACTIVE_SKILL_CONTRACTS = (
    ROOT / "CONTEXT.md",
    ROOT / "docs" / "myclaw-personal-agent-prd.md",
    ROOT / "docs" / "myclaw-runtime-contracts.md",
    ROOT / "docs" / "terminal-conversation-ui-design.md",
    ROOT / "docs" / "release-readiness.md",
)
_ACTIVE_ADRS = tuple(
    ROOT / "docs" / "adr" / name
    for name in (
        "0001-file-first-local-persistence.md",
        "0002-fixed-agent-home.md",
        "0005-store-workspace-state-in-workspace.md",
        "0007-use-host-adapters.md",
        "0008-use-workspace-session-log.md",
        "0009-active-session-snapshot-persistence.md",
        "0010-fixed-tool-catalog-and-base-tool-boundaries.md",
        "0011-use-full-screen-terminal-conversation.md",
        "0012-use-textual-for-terminal-conversation.md",
        "0015-use-session-blackboard-task-framing.md",
        "0017-use-cli-composition-root-and-session-scoped-agent-loop.md",
    )
)
_SUPERSEDED_ADRS = (
    ROOT / "docs" / "adr" / "0014-use-message-bus-agent-loop-and-agent-runner.md",
    ROOT / "docs" / "adr" / "0016-use-agent-home-skill-catalog-and-progressive-loading.md",
)
_OBSOLETE_SKILL_MARKERS = (
    "adr-0016 proposes",
    "只有内置 slash commands 进入 management port",
    "其它 `/` 开头文本作为普通用户消息发送给模型",
    "exec、web 和 workspace 外部路径按具体目标执行一次性确认。",
    "解析到 workspace 外的路径请求一次性确认。",
    "exec、web 和 workspace 外部路径才使用精确绑定的一次性确认。",
)

_ISSUE_202_ARCHITECTURE_DOCS = (
    ROOT / "CONTEXT.md",
    ROOT / "docs" / "myclaw-personal-agent-prd.md",
    ROOT / "docs" / "myclaw-runtime-contracts.md",
    ROOT / "docs" / "adr" / "0014-use-message-bus-agent-loop-and-agent-runner.md",
    ROOT / "docs" / "adr" / "0016-use-agent-home-skill-catalog-and-progressive-loading.md",
    ROOT / "docs" / "adr" / "0017-use-cli-composition-root-and-session-scoped-agent-loop.md",
    ROOT / "docs" / "cli-composition-root-implementation-plan.md",
)
_ISSUE_202_INTERIM_STATUS = "Implementation status: T1-T8 verification in progress"
_ISSUE_202_FINAL_STATUS = "Implementation status: T1-T8 complete after final verification"
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
    "tests/test_cli.py::test_cli_async_root_owns_lifetime_components_and_async_shutdown",
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
    "RuntimeSkillSnapshot",
    "build_runtime_skill_snapshot",
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
)
_ISSUE_202_FORBIDDEN_PARENT_IMPORTS = {
    "myclaw.agent": {"runtime", "workspace"},
    "myclaw.memory": {"memory_scheduler"},
}

_STANDARDS_2_3_FORBIDDEN_MODULES = (
    "myclaw.agent.repl",
    "myclaw.memory.memory_task",
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


def _tracked_markdown_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ("git", "ls-files", "-z", "--", "*.md"),
        cwd=ROOT,
    ).decode("utf-8")
    return tuple(ROOT / Path(relative) for relative in output.split("\0") if relative)


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


def _markdown_section(content: str, heading: str) -> str:
    lines = content.splitlines()
    marker = f"### {heading}"
    start = lines.index(marker) + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("### ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _glossary_definition(content: str, term: str) -> str:
    marker = f"**{term}**:\n"
    _before, found, remainder = content.partition(marker)
    assert found, term
    definition, found, _after = remainder.partition("\n_Avoid_:")
    assert found, term
    return " ".join(definition.casefold().split())


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
        "myclaw/agent/runtime.py",
        "myclaw/agent/repl.py",
        "myclaw/memory/memory_task.py",
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
                "    'myclaw.agent.runtime',\n"
                "    'myclaw.agent.repl',\n"
                "    'myclaw.memory.memory_task',\n"
                "    'myclaw.terminal.repl',\n"
                ")\n"
                "for legacy_module in legacy_modules:\n"
                "    assert importlib.util.find_spec(legacy_module) is None\n"
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


def test_user_and_release_docs_publish_the_session_log_risk_contract() -> None:
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

    for path in (ROOT / "README.md", ROOT / "docs" / "release-readiness.md"):
        content = path.read_text(encoding="utf-8").lower()
        assert all(statement in content for statement in required_contract), path


def test_active_contract_docs_do_not_claim_the_removed_runtime_log_implementation() -> None:
    active_contracts = (
        ROOT / "CONTEXT.md",
        ROOT / "docs" / "myclaw-runtime-contracts.md",
        ROOT / "docs" / "release-readiness.md",
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

    active_paths = (
        ROOT / "CONTEXT.md",
        ROOT / "README.md",
        ROOT / "docs" / "myclaw-runtime-contracts.md",
        ROOT / "docs" / "release-readiness.md",
    )
    support = "\n".join(path.read_text(encoding="utf-8").lower() for path in active_paths)
    for claim in (
        "py3-none-any",
        "windows x64",
        "currently validated",
        "macos intel",
        "apple silicon",
        "unverified",
        "no platform gate",
    ):
        assert claim in support


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


def test_current_adrs_have_unique_numbers_and_accepted_status() -> None:
    decisions = sorted((ROOT / "docs" / "adr").glob("*.md"))
    numbers = [path.name.split("-", 1)[0] for path in decisions]

    assert len(numbers) == len(set(numbers))
    assert all(_adr_status(path) == "accepted" for path in decisions)


def test_tracked_markdown_local_links_resolve() -> None:
    tracked = _tracked_markdown_paths()

    missing: list[str] = []
    for source in tracked:
        content = source.read_text(encoding="utf-8")
        for target in _markdown_link_targets(content):
            local_path = _local_markdown_path(target)
            if local_path is None:
                continue
            candidate = (source.parent / local_path).resolve()
            if not candidate.exists():
                missing.append(f"{source.relative_to(ROOT)}: {target} -> {candidate}")

    assert missing == []


def test_active_skill_docs_publish_the_accepted_routing_contract() -> None:
    skill_adr = _ACTIVE_ADRS[-1]
    active_contracts = (*_ACTIVE_SKILL_CONTRACTS, *_ACTIVE_ADRS)
    tracked = set(_tracked_markdown_paths())

    assert not [path for path in active_contracts if path not in tracked]
    assert set(_SUPERSEDED_ADRS).isdisjoint(_ACTIVE_ADRS)
    assert _adr_status(skill_adr) == "accepted"

    adr = skill_adr.read_text(encoding="utf-8").casefold()
    runtime_contract = (
        (ROOT / "docs" / "myclaw-runtime-contracts.md").read_text(encoding="utf-8").casefold()
    )
    current_skill_contract = f"{adr}\n{runtime_contract}"
    for claim in (
        "complete `skill.md` document",
        "只读取一次完整 utf-8 document",
        "loadedskill",
        "document: str",
        "skillloader",
        "def load(self) -> skillsnapshot",
    ):
        assert claim in current_skill_contract
    assert "body: str" not in runtime_contract
    assert "snapshot: skillsnapshot" not in runtime_contract

    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    glossary_contract = {
        "Skill": (
            "named, discoverable instruction package",
            "existing capabilities",
            "without registering tools or expanding permissions",
        ),
        "Skill Catalog": (
            "ordered set of valid skill metadata",
            "without loading the corresponding skill instructions",
        ),
        "Skill Invocation": (
            "selection and application",
            "foreground agent run",
            "explicitly by the user or autonomously by the model",
        ),
    }
    for term, claims in glossary_contract.items():
        definition = _glossary_definition(context, term)
        assert all(claim in definition for claim in claims), term

    runtime_contract = (ROOT / "docs" / "myclaw-runtime-contracts.md").read_text(encoding="utf-8")
    for claim in (
        "原始 `name` 不做 trim",
        "固定十个结构化 Tool schemas",
        "Tab is not intercepted",
    ):
        assert claim in runtime_contract
    assert "always_body" not in runtime_contract

    prd = (ROOT / "docs" / "myclaw-personal-agent-prd.md").read_text(encoding="utf-8")
    management_contract = _markdown_section(prd, "CLI and management").casefold()
    for claim in (
        "only management commands enter the management port",
        "exact valid skill slash invocation remains an ordinary foreground agent run",
        "unknown or non-matching slash input remains ordinary input",
    ):
        assert claim in management_contract

    tool_gateway_contract = _markdown_section(
        prd, "Tool Gateway and fail-closed security"
    ).casefold()
    for claim in (
        "`read_file`",
        "canonical agent home skill root",
        "无需 tool confirmation",
        "resolved escape",
        "仍请求一次性确认",
    ):
        assert claim in tool_gateway_contract

    schedule_contract = _markdown_section(prd, "Schedule").casefold()
    for claim in (
        "generic `read_file` path exemption",
        "不获得 skill discovery 或 invocation interface",
    ):
        assert claim in schedule_contract

    for path in active_contracts:
        content = path.read_text(encoding="utf-8").casefold()
        stale = [marker for marker in _OBSOLETE_SKILL_MARKERS if marker in content]
        assert stale == [], f"{path}: {stale}"


def test_superseded_adrs_remain_historical_and_link_the_current_authority() -> None:
    for path in _SUPERSEDED_ADRS:
        content = path.read_text(encoding="utf-8")
        assert _adr_status(path) == "accepted"
        assert "superseded by [ADR-0017]" in content
        assert "0017-use-cli-composition-root-and-session-scoped-agent-loop.md" in content

    current = (
        ROOT / "docs" / "adr" / "0017-use-cli-composition-root-and-session-scoped-agent-loop.md"
    ).read_text(encoding="utf-8")
    assert "final linearization refinement" in current
    assert "target preparation is a precondition" in current


def test_issue_202_release_closure_requires_tracked_authoritative_documents() -> None:
    tracked = {path.relative_to(ROOT).as_posix() for path in _tracked_markdown_paths()}
    required = {path.relative_to(ROOT).as_posix() for path in _ISSUE_202_ARCHITECTURE_DOCS}

    assert required <= tracked

    release = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    assert "docs/skill-module-implementation-plan.md" not in release
    for relative_path in required:
        assert relative_path in release

    plan = (ROOT / "docs" / "cli-composition-root-implementation-plan.md").read_text(
        encoding="utf-8"
    )
    assert _ISSUE_202_INTERIM_STATUS in plan or _ISSUE_202_FINAL_STATUS in plan
    assert "\u672a\u5f00\u59cb" not in plan


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
    assert _issue_202_method_names(skill_loader) == {"load"}

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
        _issue_202_ast(ROOT / "myclaw" / "memory" / "manager.py"),
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
        _issue_202_ast(ROOT / "myclaw" / "memory" / "dream.py"),
        "Dream",
    )
    assert _issue_202_parameter_names(_issue_202_function(dream, "__init__")) == (
        "self",
        "memory_manager",
        "model_router",
        "batch_size",
        "max_iterations",
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

    shutdown = next(
        node for node in ast.walk(cli_tree) if isinstance(node, ast.Try) and node.finalbody
    )
    final_tree = ast.Module(body=shutdown.finalbody, type_ignores=[])
    shutdown_events = (
        min(_issue_202_attribute_call_lines(final_tree, "management", "deactivate")),
        min(_issue_202_attribute_call_lines(final_tree, "schedule_service", "pause_and_drain")),
        min(_issue_202_attribute_call_lines(final_tree, "schedule_service", "close")),
        min(_issue_202_named_call_lines(final_tree, {"abort_loop_once", "close_loop_once"})),
        min(_issue_202_attribute_call_lines(final_tree, "dream", "close")),
        min(_issue_202_attribute_call_lines(final_tree, "router", "close")),
    )
    assert shutdown_events == tuple(sorted(shutdown_events))


def test_issue_202_release_closure_maps_real_persistence_and_architecture_nodes(
    tmp_path: Path,
) -> None:
    release = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    normalized_release = " ".join(release.split())
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

    assert all(node in release for node in evidence_nodes)

    required_claims = (
        "The Dream System Job is the only intentionally new persisted record type.",
        "The six compatibility persistence surfaces keep their current exact schemas.",
        "Dream registration creates no foreground Session or Schedule Session.",
        "tests/test_cli.py::test_cli_resume_constructor_failure_terminates_safely",
        "tests/test_cli.py::test_cli_resume_preflight_failure_terminates_safely",
        "target preparation is a precondition",
        "quiesce_for_rebind -> pause_and_drain -> current unavailable -> old abort/drain",
        "Management deactivate -> Schedule pause_and_drain + close -> Loop close/abort",
    )
    assert all(claim in normalized_release for claim in required_claims)


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
        "from myclaw.memory.memory_scheduler import MemoryTaskScheduler\n"
    )
    stale_fixture_findings = _issue_202_stale_symbol_findings(
        stale_fixture,
        "stale_fixture.py",
    )
    assert len(stale_fixture_findings) == 8

    release = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    for symbol in (*_ISSUE_202_FORBIDDEN_RUNTIME_NAMES, "read_body"):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", release):
            violations.append(f"docs/release-readiness.md: {symbol}")

    assert violations == []
    assert not (ROOT / "myclaw" / "agent" / "runtime.py").exists()
    assert not (ROOT / "myclaw" / "agent" / "workspace.py").exists()
    legacy_scheduler_module = "_".join(("memory", "scheduler"))
    assert not (ROOT / "myclaw" / "memory" / f"{legacy_scheduler_module}.py").exists()


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
        if importlib.util.find_spec(module) is not None:
            violations.append(f"{module}: deleted module is discoverable")

    assert violations == []

    conversation_summary = _issue_202_class(
        _issue_202_ast(ROOT / "myclaw" / "memory" / "conversation_summary.py"),
        "ConversationSummaryManager",
    )
    summary_init = _issue_202_direct_method(conversation_summary, "__init__")
    assert _issue_202_parameter_names(summary_init) == (
        "self",
        "provider",
        "memory_manager",
        "route_context_window",
        "route_max_output",
        "consolidation_message_threshold",
        "tools",
        "now",
        "project_messages",
    )
    assert all(default is None for default in summary_init.args.kw_defaults)

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
        _issue_202_ast(ROOT / "myclaw" / "memory" / "store.py"),
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

    terminal_design = (ROOT / "docs" / "terminal-conversation-ui-design.md").read_text(
        encoding="utf-8"
    )
    runtime_contracts = (ROOT / "docs" / "myclaw-runtime-contracts.md").read_text(encoding="utf-8")
    release_readiness = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    implementation_plan = (ROOT / "docs" / "cli-composition-root-implementation-plan.md").read_text(
        encoding="utf-8"
    )
    assert "run_repl" not in terminal_design
    assert "run_repl" not in runtime_contracts
    assert "class SummaryStore" not in runtime_contracts
    assert "class MemoryStore" not in runtime_contracts
    assert "tests/memory/test_memory_task.py" not in release_readiness
    assert "ManagementDispatcher" not in implementation_plan
    assert "current_memory_manager" not in implementation_plan
    assert "current_dream" not in implementation_plan


def test_issue_202_authoritative_documents_identify_one_current_composition_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    workspace_state_adr = (
        ROOT / "docs" / "adr" / "0005-store-workspace-state-in-workspace.md"
    ).read_text(encoding="utf-8")
    adr_0014 = (
        ROOT / "docs" / "adr" / "0014-use-message-bus-agent-loop-and-agent-runner.md"
    ).read_text(encoding="utf-8")
    adr_0016 = (
        ROOT / "docs" / "adr" / "0016-use-agent-home-skill-catalog-and-progressive-loading.md"
    ).read_text(encoding="utf-8")
    adr_0017_path = (
        ROOT / "docs" / "adr" / "0017-use-cli-composition-root-and-session-scoped-agent-loop.md"
    )
    adr_0017 = adr_0017_path.read_text(encoding="utf-8")
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    prd = (ROOT / "docs" / "myclaw-personal-agent-prd.md").read_text(encoding="utf-8")
    runtime_contract = (ROOT / "docs" / "myclaw-runtime-contracts.md").read_text(encoding="utf-8")
    terminal_design = (ROOT / "docs" / "terminal-conversation-ui-design.md").read_text(
        encoding="utf-8"
    )
    issue_195_plan = (
        ROOT / "docs" / "issue-195-terminal-commit-cancellation-fix-plan.md"
    ).read_text(encoding="utf-8")
    release_readiness = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs" / "cli-composition-root-implementation-plan.md").read_text(
        encoding="utf-8"
    )

    for stale_claim in (
        "Runtime Host",
        "Memory Task",
        "重新校验并读取完整 `SKILL.md`",
        "每个 Schedule Job 使用独立 Schedule Session",
        "<encoded_tool_call_id>",
    ):
        assert stale_claim not in readme
    assert "| CLI composition root |" in readme
    assert "Runtime Generation Skill Snapshot" in readme
    assert "Dream System Job" in readme
    assert "创建或校正 `schedule.json`" in readme
    assert (
        "Registration of the Dream System Job also creates or reconciles `schedule.json`"
        in workspace_state_adr
    )

    assert "`Tab` is not intercepted" in terminal_design
    assert (
        "Selecting any Conversation Session, including the already active Session"
        in terminal_design
    )
    assert "shared Runtime-Lifetime Message Bus" in terminal_design
    assert "Target construction or preflight failure is fatal" in terminal_design
    assert "already active Session is a no-op" not in terminal_design

    issue_195_status = issue_195_plan.splitlines()[2]
    assert "已完成" in issue_195_status
    assert "修复前实现事实" in issue_195_plan
    assert "待评审" not in issue_195_status
    assert "相比当前实现" not in plan

    current_gates = release_readiness.split("## Verification Gates", maxsplit=1)[1].split(
        "\n## ", maxsplit=1
    )[0]
    assert "1,412 passed" in current_gates
    assert "1,422 nodes total" in current_gates
    assert "1,407 passed" not in current_gates

    assert _adr_status(adr_0017_path) == "accepted"
    assert "superseded by [ADR-0017]" in adr_0014
    assert "superseded by [ADR-0017]" in adr_0016
    assert all(marker in adr_0017 for marker in ("CLI", "Agent Loop", "Runtime Generation"))
    assert all(marker in context for marker in ("Runtime Generation", "Dream"))
    assert all(marker in prd for marker in ("Message Bus", "Dream", "Agent Loop"))
    assert all(marker in runtime_contract for marker in ("D18", "Dream", "Runtime Generation"))
    for document in (adr_0017, prd, runtime_contract, plan):
        assert "final linearization refinement" in document
        assert "target preparation is a precondition" in document
        assert "quiesce_for_rebind -> pause_and_drain -> current unavailable" in document
        assert "target.start() -> publish current -> schedule_service.resume()" in document


def test_issue_202_plan_records_final_verification_only_after_all_gates() -> None:
    plan = (ROOT / "docs" / "cli-composition-root-implementation-plan.md").read_text(
        encoding="utf-8"
    )

    assert (_ISSUE_202_INTERIM_STATUS in plan) ^ (_ISSUE_202_FINAL_STATUS in plan)
    if _ISSUE_202_INTERIM_STATUS in plan:
        return

    for command in (
        "python -m pytest",
        "python -m ruff check .",
        "python -m mypy",
        "python -m build",
        "git diff --check",
    ):
        assert command in plan
    assert "docs/release-readiness.md" in plan
    assert "Verification base: clean `d60b96d1beed98b4325d2913b674be32d669adb3`" in plan

    release = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    assert "clean `d60b96d1beed98b4325d2913b674be32d669adb3`" in release
    assert "Mapped owner-node execution: 14 passed" in release
    assert "Release contract tests: " in release
