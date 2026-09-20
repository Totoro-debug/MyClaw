---
status: accepted
---

# Use Host Adapters for Portable Runtime Behavior

MyClaw has no supported-platform gate. Windows selects native Windows filesystem behavior, while other hosts attempt the POSIX filesystem behavior and fail at the operation that needs an unavailable capability. Native path conversion, object and redirection checks, containment, atomic replacement, and host-appropriate synchronization are concentrated in the host filesystem module.

Exec uses the process-lifetime PowerShell or Bash Host Adapter selected and owned as defined by [ADR-0010](0010-fixed-tool-catalog-and-base-tool-boundaries.md). That decision supersedes this ADR's former direct-Bash-only clause. Packaging emits one `py3-none-any` wheel; Windows x64 is currently validated, macOS Intel and Apple Silicon remain intended but unverified, and no formal support claim is made for other POSIX hosts.
