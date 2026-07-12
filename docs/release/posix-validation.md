# POSIX Validation

Status: **PASS**

The authoritative POSIX result is produced by the repository's permanent
[release validation workflow](../../.github/workflows/release-validation.yml) on
GitHub-hosted `ubuntu-latest`. A Windows-hosted Linux typeshed check, MSYS shell, or
an unavailable WSL registration is not accepted as POSIX execution evidence.

## Candidate

| Field | Evidence |
| --- | --- |
| Branch | `codex/b18-posix-validation` |
| Commit SHA | `5d90603136a26a656d4007446c6803abdbbc7810` |
| Workflow run | [29211545085](https://github.com/Totoro-debug/MyClaw/actions/runs/29211545085), job [86700011240](https://github.com/Totoro-debug/MyClaw/actions/runs/29211545085/job/86700011240), `success` in 1m24s |
| Runner image | GitHub-hosted `ubuntu-24.04`, image release `20260705.232` |
| Distribution | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | Linux `6.17.0-1018-azure`, x86_64 |
| Python | CPython `3.12.13` |
| pip | `26.1.2` for Python 3.12 |

## Required Gates

| Gate | Command / public seam | Required result | Current result |
| --- | --- | --- | --- |
| Offline tests | `python -m pytest -q -ra` | All non-platform tests pass; no external Provider or network call | PASS: `644 passed, 8 skipped in 22.88s` |
| POSIX process behavior | `python -m pytest -q -ra tests/test_shell_process.py tests/test_web_search.py tests/test_runtime_shutdown.py tests/test_security_shell.py` | Real Shell process, process-group descendant cleanup, timeout, cancellation, WebSearch subprocess ownership, and Runtime shutdown tests pass | PASS: `74 passed, 3 skipped in 1.86s` |
| Lint | `python -m ruff check src tests` | Pass | PASS: `All checks passed!` |
| Format | `python -m ruff format --check src tests` | Pass | PASS: `111 files already formatted` |
| Strict types | `python -m mypy src tests` | Pass under the repository's strict configuration | PASS: `Success: no issues found in 111 source files` |
| Package | `python -m build` | sdist and wheel build successfully | PASS: built `myclaw-0.1.0.tar.gz` and `myclaw-0.1.0-py3-none-any.whl` |
| Clean wheel install | Create a fresh venv, install `dist/*.whl`, and run `pip check` | Install succeeds without importing from the checkout and dependencies are consistent | PASS: wheel and dependencies installed; `No broken requirements found.` |
| CLI first start | Run installed `myclaw` with an empty temporary `HOME` | Exact configuration-gate exit, safe diagnostic, no traceback, and default `~/.myclaw/config.toml` creation | PASS: exit `2`, `config_missing`, no `Traceback`, and file assertions succeeded |
| CLI config | Run installed `myclaw config` from `workspace-验收` | Exit zero and rendered output contains `[runtime]` | PASS: command and exact grep assertion succeeded |
| Wheel import | Import from the clean venv outside the checkout | Prints `WHEEL_IMPORT_OK 0.1.0` | PASS: exact output observed |

Dependency bootstrap and clean-wheel dependency installation may use the package
index. The pytest commands themselves are the offline gates: their model, HTTP/DNS,
clock, and subprocess failure dependencies use local scripted or injected boundaries,
and they do not perform a live Provider or public-network smoke.

## Skip Policy

The full Ubuntu run must report every skip with `-ra`. The expected platform skips
are seven statically Windows-only cases plus the Windows junction-only Memory test:

- Windows Shell creation flags and Win32 handle accounting: 2.
- Windows executable lookup: 1.
- NTFS alternate streams and Windows device paths: 3.
- Windows `NUL` device-name behavior: 1.
- Windows directory-junction-only Memory case: 1.

Expected and observed full-suite total: **8 skipped**. The raw log listed exactly the
eight cases above. POSIX symlink, hard-link, real Shell, process-group descendant,
timeout, and cancellation tests were not counted as skipped. The focused process
command skipped only its three explicitly Windows-only cases and passed its other 74
tests; this proves the real POSIX Shell descendant, timeout, and cancellation nodes
executed. A Windows-only skip remains coverage separation, never evidence that the
corresponding POSIX behavior passed.

## Local Host Diagnosis

The local Windows host could not provide authoritative POSIX execution:

- The installed Store package reported WSL `2.7.8.0` and bundled kernel
  `6.18.33.1-1`, on Windows `10.0.26200.8655`. This is package metadata, not a
  running Linux kernel observation.
- `C:\Windows\System32\wsl.exe --version`, `--status`, and `--list --verbose`
  exited `1` with `系统找不到指定的路径。`.
- The WSL registry contained `ubuntu2204` (`Ubuntu 22.04`, WSL 2), but its recorded
  root at `D:\Programs\WSL\ubuntu2204` did not exist. Direct AppX `--status` produced
  no output and was terminated after 15 seconds. No distro `/etc/os-release`, kernel,
  or Linux Python process could be observed.
- Docker, Podman, nerdctl, VirtualBox, QEMU, Multipass, Rancher Desktop, and Hyper-V
  tooling were unavailable.
- Git Bash reported `MSYS_NT-10.0-26200` and used Windows process semantics/Python;
  it cannot execute the `os.name != "nt"` production branches and was rejected as a
  substitute.

No system installation or repair was attempted for validation. The GitHub-hosted
Ubuntu runner is therefore the sole POSIX authority for this release candidate.

## Platform Differences And Limits

- POSIX Shell execution uses a new session/process group and terminates descendants
  with POSIX signals; Windows uses process creation flags plus Job Objects. Both need
  their own platform report.
- POSIX symlinks and hard links are required to execute, not skip, on the Ubuntu
  runner. NTFS ADS, device names, executable lookup, Job handles, and directory
  junctions remain Windows-report responsibilities.
- The workflow validates Python 3.12, the declared minimum. It does not claim a live
  paid-Provider smoke or future third-party network availability.
- The clean-wheel smoke proves packaging, import, first-start gating, configuration
  inspection, and a Unicode Workspace path. It does not enter a paid Provider-backed
  conversation.
- The hardened workflow pins `actions/checkout` v7.0.0 and `actions/setup-python`
  v6.3.0 to full commit SHAs and disables persisted checkout credentials. The
  authoritative run completed without the earlier Node.js 20 deprecation annotation.

## Final Decision

**PASS.** Run `29211545085` checked out the exact recorded SHA, every required step
completed successfully, the full and focused skip lists matched the platform policy,
and the clean-wheel/Unicode CLI assertions passed. This is the authoritative POSIX
evidence for the B18 release candidate.
