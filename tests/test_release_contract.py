import ast
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_SKILL_PLAN = Path("docs/skill-module-implementation-plan.md")

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
        "0014-use-message-bus-agent-loop-and-agent-runner.md",
        "0015-use-session-blackboard-task-framing.md",
        "0016-use-agent-home-skill-catalog-and-progressive-loading.md",
    )
)
_OBSOLETE_SKILL_MARKERS = (
    "adr-0016 proposes",
    "只有内置 slash commands 进入 management port",
    "其它 `/` 开头文本作为普通用户消息发送给模型",
    "exec、web 和 workspace 外部路径按具体目标执行一次性确认。",
    "解析到 workspace 外的路径请求一次性确认。",
    "exec、web 和 workspace 外部路径才使用精确绑定的一次性确认。",
)

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
    relative_paths = {path.relative_to(ROOT).as_posix() for path in tracked}
    assert PROTECTED_SKILL_PLAN.as_posix() not in relative_paths

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
    assert _adr_status(skill_adr) == "accepted"

    adr = skill_adr.read_text(encoding="utf-8").casefold()
    manual_contract = next(
        paragraph for paragraph in adr.split("\n\n") if "user slash invocation" in paragraph
    )
    for claim in (
        "read and revalidate the complete `skill.md`",
        "instruction body after its frontmatter",
        "complete raw document are not projected",
    ):
        assert claim in manual_contract

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
        "`RuntimeSkillSnapshot`",
        "`catalog` 只拥有 metadata",
        "`always_loaded` 单独拥有",
        "entries: tuple[SkillMetadata, ...]",
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
