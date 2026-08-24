import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    assert all("status: accepted" in path.read_text(encoding="utf-8").lower() for path in decisions)
