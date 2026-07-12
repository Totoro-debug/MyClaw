# Windows Validation

Status: **PASS**

This report records the Windows half of the cross-platform release evidence for
GitHub issue #36. The POSIX authority is tracked separately in
[posix-validation.md](posix-validation.md).

## Candidate

| Field | Evidence |
| --- | --- |
| Product/package candidate SHA | `71cc244845e687458131a5c454411b3ddda41e5d` |
| Hardened B18 evidence SHA | `5d90603136a26a656d4007446c6803abdbbc7810` |
| Operating system | Windows 11 Pro, version `10.0.26200`, build `26200`, 64-bit, x64 |
| PowerShell | `7.6.3` |
| Build/test Python | CPython `3.12.13` |
| Build/test pip | `26.1.2` |
| Clean-venv Python | CPython `3.12.13` |
| Clean-venv pip | `25.0.1` |
| Validation root | `C:\Users\Totoro\AppData\Local\Temp\myclaw-b18-windows-20260713-055351` |

No Anthropic or OpenAI API key was present during the validation. Tests and the
package build did not call a live Provider or public-network service.

The hardened B18 commit adds only CI, documentation, and the RT-S06 regression;
`src/`, `pyproject.toml`, and the runtime package inputs are unchanged from the
product/package candidate. An independent reviewer reran the current Windows tree
at `5d90603`: the full suite passed `652` tests in 48.88s, and Ruff lint/format,
strict Mypy over 111 files, and `git diff --check` also passed. The clean-wheel
artifact hashes below continue to identify the unchanged product/package candidate.

## Windows Gates

| Gate | Command | Result |
| --- | --- | --- |
| Full offline test suite | `python -m pytest -q` | PASS: `651 passed in 58.12s` on the package candidate; `652 passed in 48.88s` on hardened B18 after adding RT-S06 |
| Lint | `python -m ruff check .` | PASS: `All checks passed!` |
| Format | `python -m ruff format --check .` | PASS: `111 files already formatted` |
| Strict types | `python -m mypy src tests` | PASS: `Success: no issues found in 111 source files` |
| Package | `python -m build --no-isolation --outdir <validation-root>\artifacts` | PASS: wheel and sdist built |
| Diff hygiene | `git diff --check` | PASS; Git only reported the existing LF-to-CRLF worktree notice for `Procedure.md` |

`--no-isolation` used the already provisioned build backend, so the packaging gate
did not bootstrap or download build requirements. The host emitted a stale
setuptools `upload_docs` entry-point warning and the setuptools disabled-test-command
deprecation warning; neither affected the successful wheel or sdist build.

## Build Artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `myclaw-0.1.0-py3-none-any.whl` | 106,382 bytes | `F9E4D7BACDA78A0D0022EB5F09F92C9017D59C5755139C89B66272195C334F2C` |
| `myclaw-0.1.0.tar.gz` | 206,316 bytes | `9F257A5CC780E3706CC756795189937D2DCC36FD85037D0EC99C2416FB291047` |

## Clean Wheel Install

The wheel was installed by absolute path into a newly created venv under the
validation root, with `PYTHONPATH` empty and the command run outside the checkout:

```powershell
python -m venv <validation-root>\venv
<validation-root>\venv\Scripts\python.exe -m pip install `
  <validation-root>\artifacts\myclaw-0.1.0-py3-none-any.whl
<validation-root>\venv\Scripts\python.exe -m pip check
```

The install used the configured package index for ordinary runtime dependency
resolution. It did not import from the repository: the checkout was absent from
`sys.path`, and `pip show myclaw` reported
`<validation-root>\venv\Lib\site-packages`. `pip check` returned
`No broken requirements found.` The installed console entry point was
`myclaw=myclaw.cli:app`.

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

Both CLI smokes used the installed `myclaw.exe`, an empty Unicode home named
`\u7528\u6237-\u9a8c\u8bc1-2`, a separate Unicode cwd, an empty `PYTHONPATH`,
`PYTHONUTF8=1`, and `PYTHONIOENCODING=utf-8`. Captured stdout and stderr were decoded
with strict UTF-8.

| Smoke | Result |
| --- | --- |
| First `myclaw` start | PASS: exit `2`, empty stderr, no traceback, and a `config_missing` message containing the correct Unicode config path |
| First-start files | PASS: `.myclaw/config.toml`, `.myclaw/memory/`, `.myclaw/sessions/`, and `.myclaw/memory/memory.md` were created |
| `myclaw config` | PASS: exit `0`, empty stderr, and the installed configuration rendered from the Unicode home |
| Secret redaction | PASS: after inserting the unique test key `b18-win-secret-DO-NOT-LEAK-7f3a9c`, `myclaw config` exited `0`, emitted `***REDACTED***`, and contained the secret in neither stdout nor stderr |

An initial non-authoritative harness run embedded literal Chinese characters in a
PowerShell-to-Python command and displayed replacement characters. The authoritative
rerun constructed the same path from ASCII Unicode escapes before process launch;
strict decoding then preserved the exact path. This was a harness encoding fault,
not a MyClaw failure.

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

PASS for the Windows portion of issue #36. No public CLI or packaging RED was found,
so validation-first TDD required no production-code change. Overall issue #36 remains
dependent on authoritative POSIX and consolidated release-readiness evidence.
