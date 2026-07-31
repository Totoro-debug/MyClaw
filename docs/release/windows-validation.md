# Windows x64 Validation

Status: **PASS**

This report records the release evidence for GitHub issues
[#56](https://github.com/Totoro-debug/myclaw/issues/56) and
[#68](https://github.com/Totoro-debug/myclaw/issues/68). It applies only to the
Windows x64 artifact built from the working tree based on commit
`29b8b1532f59099337aefd5aa44c0bcb65157d31` on 2026-07-31.

## Host

| Field | Value |
| --- | --- |
| Operating system | Windows 11 `10.0.26200`, x64 |
| PowerShell | `7.6.4` |
| Build Python | CPython `3.12.13`, 64-bit |
| Build pip | `26.1.2` |
| Validation root | `C:\Users\Totoro\AppData\Local\Temp\myclaw-release-8bd549d63ce7466f91cbfba7682f2f72` |

No Provider credential was read or used. The application tests and CLI smoke did
not contact a live Provider.

## Artifact

The build directory, distribution directory, and generated package metadata were
removed before running:

```powershell
python -m build --wheel
```

Exactly one file was emitted:

| Artifact | Size | SHA-256 | Embedded tag |
| --- | ---: | --- | --- |
| `myclaw-0.1.0-py3-none-win_amd64.whl` | 134,467 bytes | `2B821E26137996F357DDE14CE6557EE1E4F5E29AD2CB93C7AC2BC67D57C14E16` | `py3-none-win_amd64` |

Archive inspection found 73 packaged Python files. The packaged module set matched
the source tree, and no additional release artifact was present. Intermediate
`build` and `egg-info` directories were removed after inspection.

## Clean Installation

The exact wheel above was installed by absolute path into a new virtual environment
outside the checkout. Dependency resolution completed and `python -m pip check`
reported `No broken requirements found.`

The installed CLI smoke used Unicode Agent Home `home-用户` and Workspace
`workspace-验收-clean`, an empty `PYTHONPATH`, and `PYTHONNOUSERSITE=1`. First
start exited with code `2`, reported `config_missing` without a traceback, created
the default configuration, and did not create Workspace State while configuration
was gated. `myclaw config` exited with code `0`. An isolated `python -I` import
resolved `myclaw` from the clean venv's `Lib\site-packages`, not the checkout.

## Quality Gates

The authoritative commands are:

```powershell
python -m pytest -q tests/test_release_contract.py tests/test_platform_support.py tests/test_cli.py tests/test_package.py tests/test_templates.py tests/runtime_log
python -m pytest -q -ra
python -m ruff check myclaw tests
python -m ruff format --check myclaw tests
python -m mypy myclaw tests
git diff --check
```

| Gate | Result |
| --- | --- |
| Root-conflict focused suite | PASS: `5 passed in 4.01s` |
| Storage and filesystem safety suites | PASS: `231 passed in 18.34s` |
| Complete offline suite | PASS: `842 passed in 111.66s`; zero skips |
| Ruff lint | PASS: all checks passed |
| Ruff format over complete `myclaw tests` | PASS: `162 files already formatted` |
| Strict Mypy | PASS: no issues in 162 source files |
| Diff hygiene | PASS |

The complete decision is recorded in
[release readiness](../release-readiness.md).

## Boundaries

- This report validates the Windows x64 package, clean installation, package
  isolation, Unicode filesystem handling, configuration gate, and installed CLI.
- It does not claim a paid or live Provider conversation smoke.
- The artifact hash identifies this exact local candidate. A rebuild must be
  rehashed and revalidated.
