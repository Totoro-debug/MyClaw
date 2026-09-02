import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

from myclaw.agent.context import ContextBuilder

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "myclaw"


def test_retired_prompt_and_session_assembly_modules_are_absent() -> None:
    assert not (PACKAGE_ROOT / "agent" / "prompts.py").exists()
    assert not (PACKAGE_ROOT / "session" / "projection.py").exists()
    agent_prompt_module = ".".join(("myclaw", "agent", "prompts"))
    session_projection_module = ".".join(("myclaw", "session", "projection"))
    assert importlib.util.find_spec(agent_prompt_module) is None
    assert importlib.util.find_spec(session_projection_module) is None


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
        if node.level:
            retained = len(package) - node.level + 1
            base = package[: max(0, retained)]
        else:
            base = ()
        if node.module is not None:
            base = (*base, *node.module.split("."))
        module = ".".join(base)
        if module:
            imports.append((module, node.lineno))
        imports.extend(
            (".".join((*base, alias.name)), node.lineno)
            for alias in node.names
            if base or alias.name
        )
    return tuple(imports)


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
        for path in _python_files(PACKAGE_ROOT / "tools")
        for module, line in _imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]

    assert violations == []


def test_terminal_depends_on_ports_instead_of_tool_implementations() -> None:
    forbidden = {"myclaw.tools"}
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT / "terminal")
        for module, line in _imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]

    assert violations == []


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


def test_context_builder_does_not_import_model_request_runtime_boundaries() -> None:
    path = PACKAGE_ROOT / "agent" / "context.py"
    forbidden_prefixes = (
        "myclaw.provider",
        "myclaw.router",
        "myclaw.tools",
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
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
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
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
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
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
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

    required_summary_arguments = {
        "project_messages",
        "route_context_window",
        "route_max_output",
        "tools",
    }
    for method_name in ("_prepare_foreground_context", "_prepare_schedule_context"):
        summary_call = next(
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_summary_manager"
        )
        assert required_summary_arguments <= {
            keyword.arg for keyword in summary_call.keywords
        }

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_validate_always_loaded_skill_budget"
        for node in ast.walk(methods["preflight"])
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_foreground_runtime_status_input"
        for node in ast.walk(methods["_validate_always_loaded_skill_budget"])
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
    assert {
        "build_status_messages",
        "_build_status_messages_for_skills",
    } <= status_builder_calls
    assert "render_template" not in path.read_text(encoding="utf-8")


def test_runner_summary_and_dream_keep_context_builder_out_of_their_boundaries() -> None:
    paths = (
        PACKAGE_ROOT / "agent" / "runner.py",
        PACKAGE_ROOT / "memory" / "conversation_summary.py",
        PACKAGE_ROOT / "memory" / "dream.py",
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
        Path("myclaw/tools/files/__init__.py"),
        Path("myclaw/tools/files/file_tools.py"),
        Path("myclaw/tools/security.py"),
        Path("myclaw/tools/shell/__init__.py"),
        Path("myclaw/tools/shell/owned_process.py"),
        Path("myclaw/tools/shell/shell_tool.py"),
        Path("myclaw/tools/web/__init__.py"),
        Path("myclaw/tools/web/web_fetch.py"),
        Path("myclaw/tools/web/web_search.py"),
        Path("myclaw/tools/tool_artifacts.py"),
    )

    assert all(not (PROJECT_ROOT / path).exists() for path in removed)
