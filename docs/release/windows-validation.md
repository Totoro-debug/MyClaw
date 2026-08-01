# Windows x64 Validation

Status: **PASS**

This report records Windows evidence for GitHub issue
[#69](https://github.com/Totoro-debug/myclaw/issues/69). It applies to the final
universal-wheel candidate built from the clean tree at commit `2c67775` on
2026-08-02.

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
| Validation root | `C:\Users\Totoro\AppData\Local\Temp\myclaw-final-reviewed-0791068ccb4e49ce9fcfef9be821ab1d` |

No Provider credential is read or used. Application tests and CLI smoke do not contact
a live Provider.

## Artifact

The final gate built exactly one wheel with:

```powershell
python -m build --wheel --outdir <validation-root>\dist
```

| Artifact | Size | SHA-256 | Embedded tag |
| --- | ---: | --- | --- |
| `myclaw-0.1.0-py3-none-any.whl` | 137,653 bytes | `CDD56D4268F191E13F4A4CE6FBB47B7A4BB6ED8EE23D40BFEEBF5F3C03774802` | `py3-none-any` |

Archive inspection found 74 packaged Python files. The packaged module set matched the
source tree and contained the Windows and POSIX filesystem, Runtime Log lock, and owned
process tree adapters. No compiled extension, native library, or forced platform tag
was present.

## Clean Installation

The exact wheel above was installed by absolute path into a new virtual environment
outside the checkout. `python -m pip check` reported `No broken requirements found.`

The installed CLI smoke used Unicode Agent Home `home-用户\.myclaw` and Workspace
`workspace-验收-clean`, an empty `PYTHONPATH`, and `PYTHONNOUSERSITE=1`. First start
exited with code `2`, reported `config_missing` without a traceback, created the
default configuration, and did not create Workspace State before the configuration
gate. `myclaw config` exited with code `0`. An isolated `python -I` import resolved
`myclaw` from the clean venv's `Lib\site-packages`, not the checkout.

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
| Complete warning-strict offline suite | PASS: `869 passed in 178.00s`; zero skips |
| Ruff lint | PASS: all checks passed |
| Ruff format | PASS: 166 files already formatted |
| Strict Mypy | PASS: no issues in 166 source files |
| Diff hygiene | PASS |
| Universal artifact inspection | PASS: one wheel, 74 Python files, source set matched, zero native entries |
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
