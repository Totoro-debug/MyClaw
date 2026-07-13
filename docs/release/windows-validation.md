# Windows Validation

Status: **PASS**

This report records the Windows half of the cross-platform release evidence for
GitHub issue #36. The POSIX authority is tracked separately in
[posix-validation.md](posix-validation.md).

## Candidate

| Field | Evidence |
| --- | --- |
| Release candidate SHA | `31e2b17069bc54366edb6252c1a59cb2a78ed36e` |
| Operating system | Windows 11 Pro, version `10.0.26200`, build `26200`, 64-bit, x64 |
| PowerShell | `7.6.3` |
| Build/test Python | CPython `3.12.13` |
| Build/test pip | `26.1.2` |
| Clean-venv Python | CPython `3.12.13` |
| Clean-venv pip | `25.0.1` |
| Validation root | `C:\Users\Totoro\AppData\Local\Temp\myclaw-apache-final-7f69cd121f77440da3f1dadb13b89249` |

No Anthropic or OpenAI API key was present during the validation. Tests and the
package build did not call a live Provider or public-network service.

The final candidate adds the owner-selected Apache-2.0 license and PEP 639 package
metadata without changing `src/`. All test, type, build, artifact, clean-install,
and installed-CLI evidence below was rerun after that packaging change.

## Windows Gates

| Gate | Command | Result |
| --- | --- | --- |
| Full offline test suite | `python -m pytest -q -ra` | PASS: `653 passed in 83.85s` |
| Warning-strict suite | `python -X dev -W error::ResourceWarning -W error::RuntimeWarning -m pytest -q -ra` | PASS: `653 passed in 84.60s` |
| Lint | `python -m ruff check src tests` | PASS: `All checks passed!` |
| Format | `python -m ruff format --check src tests` | PASS: `111 files already formatted` |
| Strict types | `python -m mypy src tests` | PASS: `Success: no issues found in 111 source files` |
| Linux-target types | `python -m mypy --platform linux src tests` | PASS: `Success: no issues found in 111 source files` |
| Package | `python -m build` | PASS with isolated setuptools 83: wheel and sdist built |
| License metadata | Installed distribution plus direct wheel/sdist inspection | PASS: Core Metadata 2.4, `Apache-2.0`, `License-File: LICENSE`, one wheel `licenses/LICENSE`, and sdist root `LICENSE`; normalized official digest matched |
| Diff hygiene | `git diff --check` | PASS; only LF-to-CRLF worktree notices were emitted |

The project now requires `setuptools>=77` to support PEP 639. The host's setuptools
70.2 correctly cannot satisfy a no-isolation build, so the authoritative package gate
used standard isolated `python -m build` and provisioned setuptools 83.

## Build Artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `myclaw-0.1.0-py3-none-any.whl` | 112,966 bytes | `A2B8913C0B72C1E0CB42F3B8D9F3B50AB0B2BAAF6BB924F805EE5BEA6AFEAAA0` |
| `myclaw-0.1.0.tar.gz` | 215,840 bytes | `8D1262143A085B0CD8EDEF3D9BDA6D5EF16407FCBBE259C230626A2371C8033A` |

## Clean Wheel Install

The wheel was installed by absolute path into a newly created venv under the
validation root. The installed CLI smoke then ran outside the checkout with
`PYTHONPATH` empty:

```powershell
python -m venv <validation-root>\venv
<validation-root>\venv\Scripts\python.exe -m pip install `
  <checkout>\dist\myclaw-0.1.0-py3-none-any.whl
<validation-root>\venv\Scripts\python.exe -m pip check
```

The install used the configured package index for ordinary runtime dependency
resolution. It did not import from the repository: the checkout was absent from
`sys.path`, and `pip show myclaw` reported
`<validation-root>\venv\Lib\site-packages`. `pip check` returned
`No broken requirements found.` The installed console entry point was
`myclaw=myclaw.cli:app`; `pip show myclaw` reported the isolated site-packages path
and `License-Expression: Apache-2.0`.

Resolved direct runtime dependencies were:

| Distribution | Installed version |
| --- | --- |
| `aiohttp` | `3.14.1` |
| `anthropic` | `0.116.0` |
| `croniter` | `6.2.4` |
| `ddgs` | `9.14.4` |
| `jsonschema` | `4.26.0` |
| `openai` | `2.45.0` |
| `prompt-toolkit` | `3.0.52` |
| `rich` | `14.3.4` |
| `tomlkit` | `0.13.3` |
| `typer` | `0.26.8` |
| `tzlocal` | `5.4.4` |

## Installed CLI Smoke

The final CLI smokes used the installed `myclaw.exe`, an empty Unicode home named
`用户-验证-3`, a separate Unicode cwd named `workspace-验收-3`, and an empty
`PYTHONPATH`.

| Smoke | Result |
| --- | --- |
| First `myclaw` start | PASS: exit `2`, no traceback, and a `config_missing` message containing the correct Unicode config path |
| First-start files | PASS: `.myclaw/config.toml`, `.myclaw/memory/`, `.myclaw/sessions/`, and `.myclaw/memory/memory.md` were created |
| `myclaw config` | PASS: exit `0` and the installed configuration rendered from the Unicode home with `[runtime]` present |
| Secret redaction | PASS from the unchanged runtime baseline plus the final 653-test suite: the unique test key appeared only as `***REDACTED***` and never in captured output |

One discarded harness attempt used PowerShell's read-only `$HOME` automatic variable;
it was terminated during dependency installation before any CLI launch. The
authoritative rerun used a distinct `$testHome` variable and confined both `HOME` and
`USERPROFILE` to the validation root.

## Scope And Limits

- The Windows test and packaging gates were offline with respect to live Providers
  and public application network calls. The clean venv install did use package
  indexes to resolve declared third-party dependencies.
- This report validates Windows packaging, dependency resolution, clean-wheel
  isolation, first-start configuration gating, Unicode paths, config inspection,
  and secret redaction. It does not claim a paid-Provider conversation smoke.
- Windows process, NTFS, and shutdown behavior is covered by the full suite. POSIX
  process groups, signals, symlinks, and Linux packaging require the separate POSIX
  report and cannot be inferred from this host.
- The artifact hashes identify this exact local build. Rebuilding the sdist may
  produce a different archive hash if build timestamps are not normalized.

## Final Decision

PASS for the Windows portion of issue #36, including the Apache-2.0 packaging change.
The public installed-distribution test followed RED -> GREEN; no runtime production
code change was required.
