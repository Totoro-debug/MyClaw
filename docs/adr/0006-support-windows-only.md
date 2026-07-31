---
status: accepted
---

# Support Windows Only

MyClaw supports only 64-bit Windows on x86-64 (`win_amd64`) rather than maintaining POSIX, Linux, macOS, Windows x86, or Windows ARM64 behavior. Production paths, filesystem safety, process ownership, cross-process Runtime Log locking, tests, CI, packaging metadata, and release evidence use Windows x64 semantics exclusively; the CLI reports an unsupported-platform error immediately on any other operating system or process architecture. This removes parallel implementations and platform-conditioned tests from a Personal Agent intended only for Windows x64, at the cost of abandoning the repository's existing cross-platform release claim. This decision supersedes ADR-0004's POSIX `fcntl` locking and permission branches while retaining its Windows `msvcrt` locking design and all platform-independent Runtime Log behavior.

Release automation publishes only a `py3-none-win_amd64` wheel, with no platform-independent wheel or source distribution. Development installs from the source repository remain possible, but the CLI applies the same Windows x64 gate before performing any command.

Historical ADRs remain unchanged even when they describe superseded POSIX behavior. Active POSIX production branches, tests, Ubuntu CI, release-validation reports, and cross-platform support claims are removed or rewritten for Windows x64 only.
