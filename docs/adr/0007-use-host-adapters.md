---
status: accepted
---

# Use Host Adapters for Portable Runtime Behavior

MyClaw has no supported-platform gate. Windows selects native Windows filesystem behavior, while other hosts attempt the POSIX filesystem behavior and fail at the operation that needs an unavailable capability. Native path conversion, object and redirection checks, containment, atomic replacement, and host-appropriate synchronization are concentrated in the host filesystem module.

Exec launches one direct Bash subprocess with best-effort cleanup of that process. Packaging emits one `py3-none-any` wheel; Windows x64 is currently validated, macOS Intel and Apple Silicon remain intended but unverified, and no formal support claim is made for other POSIX hosts.
