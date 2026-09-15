import ast
import importlib.util
import inspect
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from myclaw.agent.context import ContextBuilder

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "myclaw"
_CLI_PATH = Path("myclaw/terminal/cli.py")
_CLI_TOOL_IMPORTS = frozenset(
    {
        ("myclaw.agent.tools.mcp_runtime", "MCPRuntimeManager"),
        ("myclaw.agent.tools.mcp_runtime", "MCPServerFailure"),
        ("myclaw.agent.tools.mcp_runtime", "MCPSnapshotReport"),
        ("myclaw.agent.tools.mcp_runtime", "MCPStartupReport"),
        ("myclaw.agent.tools.mcp_runtime", "MCPToolSnapshot"),
        ("myclaw.agent.tools.mcp_keywords", "MCPKeywordPreparer"),
        ("myclaw.agent.tools.tool_gateway", "BUILT_IN_TOOL_NAMES"),
    }
)


@dataclass(frozen=True, slots=True)
class _StaticImport:
    source_module: str
    symbol: str | None
    form: str
    line: int


def test_retired_prompt_and_session_assembly_modules_are_absent() -> None:
    assert not (PACKAGE_ROOT / "agent" / "prompts.py").exists()
    assert not (PACKAGE_ROOT / "session" / "projection.py").exists()
    assert not (PACKAGE_ROOT / "agent" / "session" / "projection.py").exists()
    agent_prompt_module = ".".join(("myclaw", "agent", "prompts"))
    assert importlib.util.find_spec(agent_prompt_module) is None
    for session_projection_module in (
        ".".join(("myclaw", "session", "projection")),
        ".".join(("myclaw", "agent", "session", "projection")),
    ):
        try:
            spec = importlib.util.find_spec(session_projection_module)
        except ModuleNotFoundError:
            spec = None
        assert spec is None


def test_agent_owned_packages_have_no_top_level_compatibility_exports() -> None:
    retired_modules = ("memory", "session", "tools")
    assert all(not (PACKAGE_ROOT / module).exists() for module in retired_modules)

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util\n"
                "modules = ('myclaw.memory', 'myclaw.session', 'myclaw.tools')\n"
                "assert all(importlib.util.find_spec(module) is None for module in modules)\n"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _imports(path: Path) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.module, node.lineno))
    return tuple(imports)


def _resolved_from_module(node: ast.ImportFrom, *, package: tuple[str, ...]) -> str:
    if node.level:
        retained = len(package) - node.level + 1
        base = package[: max(0, retained)]
    else:
        base = ()
    if node.module is not None:
        base = (*base, *node.module.split("."))
    return ".".join(base)


def _resolved_imports(
    source: str,
    *,
    package: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(source)
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _resolved_from_module(node, package=package)
        if module:
            imports.append((module, node.lineno))
        imports.extend(
            (".".join((module, alias.name)) if module else alias.name, node.lineno)
            for alias in node.names
            if module or alias.name
        )
    return tuple(imports)


def _resolved_static_imports(
    source: str,
    *,
    package: tuple[str, ...],
) -> tuple[_StaticImport, ...]:
    tree = ast.parse(source)
    imports: list[_StaticImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                _StaticImport(alias.name, None, "import", node.lineno) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            module = _resolved_from_module(node, package=package)
            imports.extend(
                _StaticImport(module, alias.name, "from", node.lineno) for alias in node.names
            )
    return tuple(imports)


def _imported_module_names(reference: _StaticImport) -> tuple[str, ...]:
    modules = [reference.source_module]
    if reference.form == "from" and reference.symbol is not None:
        modules.append(
            ".".join((reference.source_module, reference.symbol))
            if reference.source_module
            else reference.symbol
        )
    return tuple(modules)


def _is_tools_dependency(reference: _StaticImport) -> bool:
    return any(
        module == "myclaw.agent.tools" or module.startswith("myclaw.agent.tools.")
        for module in _imported_module_names(reference)
    )


def _terminal_tool_import_violations(
    sources: Mapping[Path, str],
    *,
    allowed_symbols: Mapping[Path, frozenset[tuple[str, str]]],
) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        allowed_for_path = allowed_symbols.get(path, frozenset())
        for reference in _resolved_static_imports(
            sources[path],
            package=tuple(path.parent.parts),
        ):
            if not _is_tools_dependency(reference):
                continue
            is_allowed = (
                reference.form == "from"
                and reference.symbol is not None
                and (reference.source_module, reference.symbol) in allowed_for_path
            )
            if is_allowed:
                continue
            imported = reference.source_module
            if reference.form == "from" and reference.symbol is not None:
                imported = f"{imported}.{reference.symbol}" if imported else reference.symbol
            violations.append(f"{path}:{reference.line} imports {imported}")
    return tuple(violations)


def _retired_mcp_runtime_import_violations(sources: Mapping[Path, str]) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        for reference in _resolved_static_imports(
            sources[path],
            package=tuple(path.parent.parts),
        ):
            imported_modules = _imported_module_names(reference)
            if not any(
                module == "myclaw.mcp_runtime" or module.startswith("myclaw.mcp_runtime.")
                for module in imported_modules
            ):
                continue
            violations.append(f"{path}:{reference.line} imports {imported_modules[-1]}")
    return tuple(violations)


def _is_blackboard_module(module: str) -> bool:
    return module == "myclaw.agent.blackboard" or module.startswith("myclaw.agent.blackboard.")


@pytest.mark.parametrize(
    "source",
    [
        "import myclaw.agent.blackboard",
        "from myclaw.agent.blackboard import Blackboard",
        "from myclaw.agent import blackboard",
        "from .blackboard import Blackboard",
        "from . import blackboard",
        "def load():\n    import myclaw.agent.blackboard",
        "if TYPE_CHECKING:\n    from . import blackboard",
    ],
)
def test_import_scanner_resolves_blackboard_dependency_forms(source: str) -> None:
    assert any(
        _is_blackboard_module(module)
        for module, _ in _resolved_imports(source, package=("myclaw", "agent"))
    )


def test_production_code_does_not_import_removed_contracts_package() -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT)
        for module, line in _imports(path)
        if module == "myclaw.contracts" or module.startswith("myclaw.contracts.")
    ]

    assert violations == []


@pytest.mark.parametrize("root", [PACKAGE_ROOT / "utils", PACKAGE_ROOT / "errors.py"])
def test_foundation_modules_do_not_import_domain_modules(root: Path) -> None:
    files = (root,) if root.is_file() else _python_files(root)
    allowed = {"myclaw.errors", "myclaw.utils"}
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in files
        for module, line in _imports(path)
        if module.startswith("myclaw.")
        and not any(module == prefix or module.startswith(f"{prefix}.") for prefix in allowed)
    ]

    assert violations == []


def test_tools_do_not_depend_on_provider() -> None:
    forbidden = {"myclaw.provider"}
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT / "agent" / "tools")
        for module, line in _imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]

    assert violations == []


def test_terminal_depends_on_ports_instead_of_tool_implementations() -> None:
    sources = {
        path.relative_to(PROJECT_ROOT): path.read_text(encoding="utf-8")
        for path in _python_files(PACKAGE_ROOT / "terminal")
    }

    violations = _terminal_tool_import_violations(
        sources,
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


@pytest.mark.parametrize(("module", "symbol"), sorted(_CLI_TOOL_IMPORTS))
def test_terminal_tool_import_checker_allows_each_cli_symbol(
    module: str,
    symbol: str,
) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: f"from {module} import {symbol}"},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


@pytest.mark.parametrize(
    "source",
    [
        "from myclaw.agent.tools.mcp_runtime import MCPRuntimeManager as Manager",
        "from ..agent.tools.mcp_runtime import MCPRuntimeManager",
        "from ..agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES as BUILT_INS",
    ],
)
def test_terminal_tool_import_checker_resolves_allowed_aliases_and_relative_imports(
    source: str,
) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: source},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


def test_terminal_tool_import_checker_retains_original_symbol_form_and_line() -> None:
    references = _resolved_static_imports(
        "\nfrom myclaw.agent.tools.mcp_runtime import MCPRuntimeManager as Manager",
        package=("myclaw", "terminal"),
    )

    assert references == (
        _StaticImport("myclaw.agent.tools.mcp_runtime", "MCPRuntimeManager", "from", 2),
    )


@pytest.mark.parametrize(
    "path",
    [
        Path("myclaw/terminal/conversation.py"),
        Path("myclaw/terminal/process_entry.py"),
        Path("myclaw/terminal/internal/loader.py"),
    ],
)
def test_terminal_tool_import_checker_rejects_cli_symbols_outside_cli(path: Path) -> None:
    violations = _terminal_tool_import_violations(
        {path: "from myclaw.agent.tools.mcp_runtime import MCPRuntimeManager"},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations


@pytest.mark.parametrize(
    "source",
    [
        "from myclaw.agent.tools.tool_gateway import ToolGateway",
        "from myclaw.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES, ToolGateway",
        "from myclaw.agent.tools.mcp_runtime import allocate_mcp_tool_name",
        "import myclaw.agent.tools.mcp_runtime",
        "import myclaw.agent.tools.mcp_runtime as runtime",
        "from myclaw.agent.tools import mcp_runtime",
        "from myclaw.agent.tools.mcp_runtime import *",
        "from myclaw.agent.tools.tool_gateway import ToolGateway as Gateway",
        "from ..agent.tools.tool_gateway import ToolGateway",
        "def load():\n    from myclaw.agent.tools.mcp_runtime import allocate_mcp_tool_name",
        "if TYPE_CHECKING:\n    from myclaw.agent.tools.mcp_runtime import allocate_mcp_tool_name",
    ],
)
def test_terminal_tool_import_checker_rejects_unapproved_tool_imports(source: str) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: source},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio",
        "from myclaw.management.service import ManagementViewService",
        "from .conversation import TerminalConversationApp",
    ],
)
def test_terminal_tool_import_checker_allows_non_tool_dependencies(source: str) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: source},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


def test_retired_mcp_runtime_export_is_absent() -> None:
    assert not (PACKAGE_ROOT / "mcp_runtime.py").exists()

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; print(importlib.util.find_spec('myclaw.mcp_runtime'))",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert probe.stdout.strip() == "None"


@pytest.mark.parametrize(
    "source",
    [
        "import myclaw.mcp_runtime",
        "from myclaw.mcp_runtime import MCPRuntimeManager",
        "from myclaw import mcp_runtime",
        "from ..mcp_runtime import MCPRuntimeManager",
        "from .. import mcp_runtime",
    ],
)
def test_retired_mcp_runtime_import_checker_covers_import_forms(source: str) -> None:
    violations = _retired_mcp_runtime_import_violations({_CLI_PATH: source})

    assert violations


def test_production_code_does_not_import_retired_mcp_runtime_export() -> None:
    sources = {
        path.relative_to(PROJECT_ROOT): path.read_text(encoding="utf-8")
        for path in _python_files(PACKAGE_ROOT)
    }

    violations = _retired_mcp_runtime_import_violations(sources)

    assert violations == ()


def test_context_builder_constructor_owns_only_context_dependencies() -> None:
    parameters = inspect.signature(ContextBuilder.__init__).parameters

    assert tuple(parameters) == (
        "self",
        "workspace",
        "timezone_name",
        "agent_home",
        "memory_manager",
        "skill_loader",
    )
    assert parameters["agent_home"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["memory_manager"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["skill_loader"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "clock" not in parameters
    assert "tool_gateway" not in parameters
    assert "tool_schema" not in parameters
    assert "router" not in parameters
    assert "model_router" not in parameters


def test_mcp_keyword_module_does_not_import_private_configuration_implementation() -> None:
    path = PACKAGE_ROOT / "agent" / "tools" / "mcp_keywords.py"
    source = path.read_text(encoding="utf-8")
    private_configuration_imports = [
        reference
        for reference in _resolved_static_imports(
            source,
            package=("myclaw", "agent", "tools"),
        )
        if reference.source_module == "myclaw.config.config"
        and reference.symbol is not None
        and reference.symbol.startswith("_")
    ]

    assert private_configuration_imports == []


def test_context_builder_does_not_import_model_request_runtime_boundaries() -> None:
    path = PACKAGE_ROOT / "agent" / "context.py"
    forbidden_prefixes = (
        "myclaw.provider",
        "myclaw.router",
        "myclaw.agent.tools",
    )
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for module, line in _imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
    ]

    assert violations == []


def test_agent_loop_does_not_retain_a_title_prompt_outside_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    retained_title_prompts = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_title_prompt"
    ]

    assert retained_title_prompts == []


def test_agent_loop_delegates_foreground_context_construction_to_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    agent_loop = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    methods = {
        node.name: node
        for node in agent_loop.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }

    def calls_builder(method_name: str, builder_method: str) -> bool:
        method = methods[method_name]
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == builder_method
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_context_builder"
            for node in ast.walk(method)
        )

    assert "render_template" not in source
    assert "build_messages" not in source
    assert "_project_foreground_messages" not in source
    assert "_project_foreground_summary_messages" not in methods
    assert calls_builder("_prepare_foreground_context", "build_foreground_messages")


def test_agent_loop_delegates_schedule_context_construction_to_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    agent_loop = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    prepare_schedule = next(
        node
        for node in agent_loop.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_prepare_schedule_context"
    )

    assert hasattr(ContextBuilder, "build_schedule_messages")
    assert "_project_schedule_messages" not in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_schedule_messages"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_context_builder"
        for node in ast.walk(prepare_schedule)
    )


def test_agent_loop_request_paths_stay_inside_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    agent_loop = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    methods = {
        node.name: node
        for node in agent_loop.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    expected_builder_calls = {
        "_prepare_foreground_context": "build_foreground_messages",
        "_prepare_schedule_context": "build_schedule_messages",
        "_router_stream_title": "build_title_messages",
    }
    for method_name, builder_method in expected_builder_calls.items():
        method = methods[method_name]
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == builder_method
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_context_builder"
            for node in ast.walk(method)
        )

    required_compaction_arguments = {
        "project_messages",
        "route_context_window",
        "route_max_output",
        "tools",
    }
    for method_name in ("_prepare_foreground_context", "_prepare_schedule_context"):
        compaction_call = next(
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_compactor"
        )
        assert required_compaction_arguments <= {keyword.arg for keyword in compaction_call.keywords}

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_validate_model_context_budget"
        for node in ast.walk(methods["preflight"])
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_foreground_runtime_status_input"
        for node in ast.walk(methods["_validate_model_context_budget"])
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_foreground_runtime_status_input"
        for node in ast.walk(methods["runtime_status_input"])
    )
    status_helper = functions["_foreground_runtime_status_input"]
    status_builder_calls = {
        node.func.attr
        for node in ast.walk(status_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "context_builder"
    }
    assert "build_status_messages" in status_builder_calls
    removed_builder_methods = {
        "_build_status_messages_for_skills",
        "_foreground_system_prompt_for_skills",
    }
    assert all(not hasattr(ContextBuilder, name) for name in removed_builder_methods)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in removed_builder_methods
        for node in ast.walk(tree)
    )
    assert "skill_state" not in {argument.arg for argument in status_helper.args.kwonlyargs}
    assert "render_template" not in path.read_text(encoding="utf-8")


def test_runner_summary_and_dream_keep_context_builder_out_of_their_boundaries() -> None:
    paths = (
        PACKAGE_ROOT / "agent" / "runner.py",
        PACKAGE_ROOT / "agent" / "memory" / "conversation_compactor.py",
        PACKAGE_ROOT / "agent" / "memory" / "dream.py",
    )
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in paths
        for module, line in _imports(path)
        if module == "myclaw.agent.context" or module.startswith("myclaw.agent.context.")
    ]
    assert violations == []


def test_agent_modules_do_not_depend_on_terminal_presentation() -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT / "agent")
        for module, line in _imports(path)
        if module == "myclaw.terminal" or module.startswith("myclaw.terminal.")
    ]

    assert violations == []


def test_terminal_conversation_lifecycle_has_no_business_lifecycle_calls() -> None:
    path = PACKAGE_ROOT / "terminal" / "conversation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    app = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerminalConversationApp"
    )
    lifecycle_methods = {"__init__", "on_mount", "on_unmount", "rebind_agent_loop"}
    business_calls = {
        "abort",
        "abort_and_wait",
        "close",
        "drain_inbound",
        "drain_outbound",
        "reset",
        "start",
    }
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{call.lineno} calls {call.func.attr}"
        for method in app.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name in lifecycle_methods
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in business_calls
    ]

    assert violations == []


def test_package_initializers_do_not_create_aggregate_import_entries() -> None:
    violations: list[str] = []
    for path in _python_files(PACKAGE_ROOT):
        if path.name != "__init__.py" or path.parent == PACKAGE_ROOT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            violations.extend(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} imports {module}"
                for module in modules
                if module == "myclaw" or module.startswith("myclaw.")
            )

    assert violations == []


def test_host_selection_is_confined_to_the_workspace_filesystem_adapter() -> None:
    expected = {Path("myclaw/utils/host_filesystem.py")}
    actual = {
        path.relative_to(PROJECT_ROOT)
        for path in _python_files(PACKAGE_ROOT)
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "name"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        )
    }

    assert actual == expected


def test_superseded_tool_modules_are_absent() -> None:
    removed = (
        Path("myclaw/agent/tools/files/__init__.py"),
        Path("myclaw/agent/tools/files/file_tools.py"),
        Path("myclaw/agent/tools/security.py"),
        Path("myclaw/agent/tools/shell/__init__.py"),
        Path("myclaw/agent/tools/shell/owned_process.py"),
        Path("myclaw/agent/tools/shell/shell_tool.py"),
        Path("myclaw/agent/tools/web/__init__.py"),
        Path("myclaw/agent/tools/web/web_fetch.py"),
        Path("myclaw/agent/tools/web/web_search.py"),
        Path("myclaw/agent/tools/tool_artifacts.py"),
    )

    assert all(not (PROJECT_ROOT / path).exists() for path in removed)
