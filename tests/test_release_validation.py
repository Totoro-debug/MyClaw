from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

import scripts.release_validation as release_validation
from scripts.release_validation import (
    COVERAGE_RULES,
    REQUIRED_WINDOWS_ALTERNATIVE_NODES,
    CoverageEvidence,
    CoverageRule,
    ReleasePhase,
    build_coverage_evidence,
)


def test_windows_release_phases_are_explicit() -> None:
    assert tuple(ReleasePhase) == (
        "coverage",
        "host-integration",
        "quality",
        "artifact-smoke",
        "all",
    )


def test_coverage_rules_are_quantified_and_fail_closed() -> None:
    assert {rule.name for rule in COVERAGE_RULES} >= {
        "shell-selection",
        "whitelist-direct-fixtures",
        "dynamic-complex",
        "identity",
        "path-edges",
        "catastrophic",
        "inspector-failures",
        "full-access-dynamic",
        "file-read",
        "file-write",
        "web",
        "schedule",
        "mcp",
        "gateway-hard-errors",
        "web-redirect-rebinding",
        "textual-modal",
        "dream-exemption",
        "catalog-stability",
        "persistence-schema",
    }
    assert all(not hasattr(rule, "multiplier") for rule in COVERAGE_RULES)
    assert all(
        pattern.startswith("^") and pattern.endswith("$")
        for rule in COVERAGE_RULES
        for pattern in rule.patterns
    )
    failing = CoverageEvidence(
        collected_nodes=("tests/example.py::test_case",),
        counts={rule.name: 0 for rule in COVERAGE_RULES},
    )
    with pytest.raises(AssertionError, match="shell-selection"):
        failing.assert_minimums()


def test_coverage_uses_distinct_explicit_pytest_nodes_and_fails_on_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pattern = release_validation._node_pattern("tests/example.py", "test_matrix")
    monkeypatch.setattr(
        release_validation,
        "COVERAGE_RULES",
        (CoverageRule("explicit-matrix", 2, (pattern,)),),
    )
    first = "tests/example.py::test_matrix[read-only]"
    second = "tests/example.py::test_matrix[full-access]"

    evidence = build_coverage_evidence(
        (first, first, second, "tests/example.py::test_matrix_renamed[full-access]")
    )

    assert evidence.counts == {"explicit-matrix": 2}
    assert evidence.details["explicit-matrix"]["matched_nodes"] == [first, second]
    with pytest.raises(AssertionError, match="observed 1; minimum is 2"):
        build_coverage_evidence((first,))


def test_windows_release_entry_is_documented_in_readme() -> None:
    readme = Path(__file__).parents[1] / "README.md"

    assert "python scripts/release_validation.py --phase all" in readme.read_text(encoding="utf-8")


def test_workflow_runs_the_single_windows_all_phase_gate() -> None:
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "release-validation.yml"
    document = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert set(document["jobs"]) == {"windows-release"}
    job = document["jobs"]["windows-release"]
    assert job["runs-on"] == "windows-latest"
    commands = "\n".join(step.get("run", "") for step in job["steps"] if isinstance(step, dict))
    assert "python scripts/release_validation.py --phase all" in " ".join(commands.split())


def test_skip_classification_is_bound_to_exact_node_and_reason() -> None:
    accepted = {
        "nodeid": (
            "tests/tools/core/test_exec_bash_policy.py::"
            "test_real_posix_bash_inspect_policy_execute_smoke"
        ),
        "message": "requires a real POSIX production host",
    }

    assert release_validation._classify_skip(accepted) == "waived-posix-host-scope"
    assert (
        release_validation._classify_skip(
            {
                **accepted,
                "nodeid": "tests/example.py::test_real_posix_bash_inspect_policy_execute_smoke",
            }
        )
        == "unclassified"
    )
    assert (
        release_validation._classify_skip(
            {**accepted, "message": "requires a real POSIX production host for another feature"}
        )
        == "unclassified"
    )


def test_windows_skip_alternatives_are_explicit_full_suite_nodes() -> None:
    assert len(REQUIRED_WINDOWS_ALTERNATIVE_NODES) >= 10
    assert all(
        node.startswith("tests/") and ".py::test_" in node
        for node in REQUIRED_WINDOWS_ALTERNATIVE_NODES
    )


def test_command_failure_includes_sanitized_process_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=9,
            stdout="standard output",
            stderr="standard error",
        ),
    )

    with pytest.raises(RuntimeError, match=r"standard output.*standard error"):
        release_validation._run_command(("tool", "argument"))


def test_command_timeout_is_a_nonzero_gate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd=("tool",), timeout=7)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="timed out after 7s"):
        release_validation._run_command(("tool",), timeout=7)


def test_coverage_evidence_is_json_serializable() -> None:
    evidence = CoverageEvidence(
        collected_nodes=("tests/example.py::test_case",),
        counts={"shell-selection": 11},
    )
    payload = evidence.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["collected_nodes"] == ["tests/example.py::test_case"]
