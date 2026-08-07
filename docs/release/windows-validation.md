# Windows x64 Validation

Status: **PASS**

This report records the Windows x64 evidence for GitHub issue
[#117](https://github.com/Totoro-debug/myclaw/issues/117), the end-to-end release
gate for parent issue [#104](https://github.com/Totoro-debug/myclaw/issues/104).
It supersedes the older Session-only count while retaining the same host-adapter
and pure-Python packaging contract.

Windows x64 is the currently validated environment. macOS Intel and Apple Silicon
remain intended compatibility targets but are unverified. This report is not native
macOS evidence and makes no formal Linux support claim. MyClaw has no platform gate;
other hosts attempt their selected host adapter when a capability is used.

## Host

| Field | Value |
| --- | --- |
| Operating system | Windows 11 `10.0.26200`, x64 |
| PowerShell | `7.6.4` |
| Build Python | CPython `3.12.13`, 64-bit |
| Build pip | `26.1.2` |
| Validation root | `C:\Users\Totoro\AppData\Local\Temp\myclaw-issue117-validation` |

No Provider credential is read or used. Runtime composition and CLI smoke use fake
offline Providers.

## Schedule Acceptance

The highest acceptance seam is Runtime composition plus the Conversation Port. The
following public-interface tests are the issue-117 evidence map:

| Acceptance area | Public evidence | Result |
| --- | --- | --- |
| Natural-language add/list/remove, approval, decline, stable JSON, dispatcher wake | `tests/scheduling/test_schedule_runtime.py` and `tests/scheduling/test_schedule_service.py` | PASS |
| at auto-delete with retained Schedule Session and resume exclusion | `test_runtime_dispatcher_wakes_for_due_at_job_and_keeps_schedule_session_out_of_resume` | PASS |
| every overlap, Cron DST gap/overlap, different-Job concurrency, foreground concurrency | `tests/scheduling/test_schedule_service.py` | PASS |
| Confirmation waiting cancellation, accepted mutation cancellation, Runtime shutdown boundaries | `test_runtime_confirmation_cancellation_preserves_declined_and_accepted_boundaries`, `tests/agent/test_conversation_port.py`, `tests/test_runtime_shutdown.py` | PASS |
| Atomic replacement failure, fault latch, restart from the last complete document | `tests/scheduling/test_schedule_store.py`, `tests/scheduling/test_schedule_service.py` | PASS |
| Corrupt startup fails before logs or schedulers and preserves the file | `tests/test_cli.py`, `tests/scheduling/test_schedule_store.py` | PASS |
| Schedule Summary stream, Summary Cursor pre-advance, and later-run memory cache | `test_runtime_schedule_summary_flows_through_memory_to_a_later_schedule_run`, `tests/memory/test_memory_task.py` | PASS |
| Schedule Session, Artifact, Session Log, and resume isolation | `tests/sessions/test_session_resume.py`, `tests/tools/test_tool_artifacts.py`, `tests/test_session_log.py` | PASS |
| Schedule outcomes stay in Schedule Sessions without Agent Events or notifications | `tests/scheduling/test_schedule_runtime.py`, `tests/scheduling/test_schedule_service.py` | PASS |

The focused composition command is:

```powershell
python -W error -m pytest -q -ra tests/scheduling/test_schedule_runtime.py
```

Result: `4 passed`.

## Artifact

The build gate must remove ignored build outputs before creating the wheel. Without
that step, setuptools can reuse stale files from a previous checkout and publish
deleted Scheduled Work modules. Run from the repository root:

```powershell
$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
foreach ($name in @("build", "dist", "myclaw.egg-info")) {
    $path = Join-Path $repo $name
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
New-Item -ItemType Directory -Force .tmp-runtime-tests\issue-117-dist | Out-Null
python -m build --wheel --no-isolation --outdir .tmp-runtime-tests\issue-117-dist
```

| Artifact | Size | SHA-256 | Embedded tag |
| --- | ---: | --- | --- |
| `myclaw-0.1.0-py3-none-any.whl` | 146,408 bytes | `3DCD69C29BEF975827B559989885CF6893BB484FE99AC13DF6836B8C000517FA` | `py3-none-any` |

The clean archive contains 73 packaged Python files and 14 template files. Its
module set matches the source tree, contains no `turn.py`, `scheduled_work*`,
`background_coordination.py`, or other removed Scheduled Work files, and contains
no compiled extension or native library.

## Quality Gates

The authoritative issue-117 commands are:

```powershell
python -W error -m pytest -q -ra
python -m ruff check myclaw tests
python -m mypy myclaw tests
git diff --check
```

| Gate | Result |
| --- | --- |
| Complete warning-strict offline suite | PASS: `901 passed`, `2 skipped` because this host lacks the privilege required for two symbolic-link tests |
| Repository Ruff lint | PASS: all checks passed |
| Strict Mypy | PASS: no issues found |
| Diff hygiene | PASS |
| Clean universal-wheel build and archive inspection | PASS: one wheel, 73 Python files, 14 templates, zero removed Schedule modules |

`ruff format --check myclaw tests` is not part of the issue-117 gate. On this
checkout it reports 37 pre-existing formatting differences; the new acceptance
test file is formatted and passes the focused format check. No unrelated formatting
churn was introduced.

## Clean Installation

The final wheel must be installed by absolute path into a new virtual environment
outside the checkout with `PYTHONNOUSERSITE=1`, followed by:

```powershell
python -m pip check
python -I -c "import myclaw; print(myclaw.__file__)"
myclaw config
```

The installed CLI smoke must use a Unicode Agent Home and Workspace, report
`config_missing` without a traceback on first start, and leave Workspace State
uninitialized for the management-only `config` command.

The install smoke must resolve `myclaw` from the clean environment, report no broken
requirements, keep the source checkout out of `sys.path`, and leave Workspace State
uninitialized for the management-only `config` command. The issue-117 smoke passed:
`pip check` reported `No broken requirements found.`, isolated import resolved from
the venv `site-packages`, and `myclaw config` returned `0`.

## Boundaries

- Fake adapters and Linux-platform static typing do not constitute native macOS validation.
- Native macOS CI and manual macOS validation remain outstanding.
- The artifact hash identifies this exact local candidate; any rebuild must be rehashed.
- File-first persistence and Session Logs remain uncoordinated across Runtime processes.
- Ordinary Session persistence has no acknowledgement or failure logging, and Summary can diverge from `last_consolidated` after a crash.
- Existing Session schemas are unsupported; no migration or version dispatch is provided.
- Shell command policy and owned-process cleanup are not an operating-system filesystem or network sandbox.

The complete decision and evidence boundary are recorded in
[release readiness](../release-readiness.md).
