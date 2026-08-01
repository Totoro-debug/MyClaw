---
status: accepted
---

# Use Host Adapters for Portable Runtime Behavior

MyClaw uses host-native behavior without a supported-platform allowlist. Windows
selects the Windows adapters; every other host attempts the POSIX adapters and lets
missing operating-system capabilities fail at the operation that needs them. This
decision supersedes ADR-0006 while retaining that document as the historical record
of the former Windows-only release.

Operating-system differences are concentrated behind three deep modules: filesystem
operations, Runtime Log locking, and owned process tree execution. The filesystem
module owns native I/O paths, object-type and redirection checks, containment,
create-only publication, atomic replacement, and host-appropriate synchronization.
Runtime Log locking uses `msvcrt` on Windows and `fcntl` on POSIX behind one lock
interface. Shell execution uses Windows Job Objects or a POSIX process session and
process group behind one owned-process interface. These adapters use only the Python
standard library.

Workspace identity is the normalized absolute path under the current host's native
path semantics. Workspace State remains `<workspace>/.myclaw/` with unchanged record
formats and lifecycle. Copying a complete Workspace carries its Workspace State;
Agent Home remains local to the operating-system account and continues to own only
User Configuration and Runtime Logs.

The installed `myclaw` command enters the Typer application directly. Packaging emits
one pure-Python `py3-none-any` wheel and performs no operating-system version,
architecture, or platform-tag gate at startup.

Windows x64 is the currently validated environment. macOS Intel and Apple Silicon are
intended compatibility targets but remain unverified until the same pytest and
installed-wheel gates run natively there. The POSIX adapter may be attempted on Linux
and other POSIX hosts, but this decision makes no formal support claim for them.
