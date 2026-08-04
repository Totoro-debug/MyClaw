# Windows x64 Validation

Status: **PASS**

This report records Windows evidence for GitHub issue
[#92](https://github.com/Totoro-debug/myclaw/issues/92). It applies to the
active-Session architecture universal-wheel candidate built on 2026-08-04.

Windows x64 is the currently validated environment. macOS Intel and Apple Silicon are
intended compatibility targets but remain unverified. This report is not native macOS
evidence and makes no formal Linux support claim. MyClaw has no platform gate, so other
hosts attempt their selected adapter when a capability is used.

## Host

| Field | Value |
| --- | --- |
| Operating system | Windows 11 `10.0.26200`, x64 |
| PowerShell | `7.6.4` |
| Build Python | CPython `3.12.13`, 64-bit |
| Build pip | `26.1.2` |
| Validation root | `C:\Users\Totoro\AppData\Local\Temp\myclaw-issue92-review-e4fddfe9` |

No Provider credential is read or used. Application tests and CLI smoke do not contact
a live Provider.

## Artifact

The final gate used `<validation-root>\build-venv\Scripts\python.exe`. With that
environment active, the equivalent module command that built exactly one wheel was:

```powershell
python -m build --wheel --no-isolation --outdir <validation-root>\dist
```

| Artifact | Size | SHA-256 | Embedded tag |
| --- | ---: | --- | --- |
| `myclaw-0.1.0-py3-none-any.whl` | 128,811 bytes | `89B89DB5B123F9135065A3F9F930F12208A9FCDEF37E67F16F0D9A5009F1B6FD` | `py3-none-any` |

Archive inspection found 72 packaged Python files and 14 template files. The packaged
module set matched the source tree and contained the Windows and POSIX filesystem and
owned-process adapters.
It contained the Loguru dependency metadata and no obsolete Runtime Log or lock module.
No compiled extension, native library, or forced platform tag was present.

## Clean Installation

The exact wheel above was installed by absolute path into a new virtual environment
outside the checkout. `python -m pip check` reported `No broken requirements found.`

The installed CLI smoke used Unicode Agent Home `home-用户\.myclaw` and Workspace
`workspace-验收-clean`, an empty `PYTHONPATH`, and `PYTHONNOUSERSITE=1`. First start
exited with code `2`, reported `config_missing` without a traceback, created the
default configuration, and did not create Workspace State before the configuration
gate. `myclaw config` exited with code `0`. An isolated `python -I` import resolved
`myclaw` from the clean venv's `Lib\site-packages`, not the checkout.

## Session and Host-Filesystem Windows Evidence

The active Session public-interface suite covered complete compact UTF-8 JSONL
replacement, snapshot freeze and call-order completion, lazy materialization of empty
Sessions, silent ordinary background failure, cleanup of queued background persistence
before shutdown, three-attempt shutdown retry with 100 ms/200 ms delays, and final
failure swallowing. The host filesystem suite exercised the atomic replacement seam and
injected synchronization failures. The combined Windows-focused command was:

```powershell
python -W error -m pytest -q -ra tests/sessions/test_session.py tests/test_host_filesystem.py tests/test_windows_filesystem.py tests/test_runtime_shutdown.py
```

Result: `72 passed`.

The native Session Log contract suite creates a Windows Junction at `.myclaw\logs`,
confirms its Reparse Point is rejected, and verifies that no file is written through
the redirect while Session work continues. A hard-linked active Session Log is also
rejected without changing either link's bytes, and the next clean Session context
successfully retries activation.

The rotation test reaches exactly 10,485,760 bytes without rotating, confirms that
the next record rotates, then performs two further rotations. The active file remains
present and only the newest history file survives. Separate injected `logger.add`,
opener, write, and rotation failures leave the business result unchanged. Consecutive
activation failures emit one basic diagnostic, a successful activation resets that
latch, and a later failure is reported again.

Native `Get-Acl` probes confirm that creating and writing the logs directory does not
change the Workspace State ACL, and that both `logs` and the active file keep Windows
ACL inheritance enabled with inherited access rules. This matches the Workspace State
contract: MyClaw preserves the Workspace's inherited DACL rather than replacing it
with a private owner-only DACL on Windows.

## Quality Gates

The authoritative commands are:

```powershell
python -W error -m pytest -q -ra
python -m ruff check myclaw tests
python -m ruff format --check myclaw tests
python -m mypy myclaw tests
git diff --check
```

| Gate | Result |
| --- | --- |
| Complete warning-strict offline suite | PASS: `726 passed`; zero skips |
| Ruff lint | PASS: all checks passed |
| Ruff format | PASS: 153 files already formatted |
| Strict Mypy | PASS: no issues in 153 source files |
| Diff hygiene | PASS |
| Universal artifact inspection | PASS: one wheel, 72 Python files, 14 templates, source set matched, zero native entries |
| Clean wheel installation and dependency check | PASS |
| Installed CLI Unicode smoke and isolated import | PASS |

## Boundaries

- Fake adapters and Linux-platform static typing do not constitute native macOS
  validation.
- Native macOS CI and manual macOS validation remain out of scope.
- The artifact hash identifies the exact final local candidate; any rebuild must be
  rehashed and revalidated.

The complete decision and evidence boundary are recorded in
[release readiness](../release-readiness.md).
