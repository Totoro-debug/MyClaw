import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "myclaw"


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


@pytest.mark.parametrize(
    "path",
    [PACKAGE_ROOT / "agent" / "events.py", PACKAGE_ROOT / "terminal" / "repl.py"],
)
def test_user_facing_agent_event_boundary_does_not_import_tool_confirmation(path: Path) -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for module, line in _imports(path)
        if module == "myclaw.tools.confirmation"
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


def test_host_selection_is_confined_to_the_three_native_deep_modules() -> None:
    expected = {
        Path("myclaw/tools/shell/owned_process.py"),
        Path("myclaw/utils/host_filesystem.py"),
    }
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
