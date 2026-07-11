# Bound first-version Shell by Workspace cwd and permission, not an OS sandbox

The first version requires every Shell `cwd` to resolve within the current Workspace. It automatically allows only five exact, read-only command shapes: `pwd`, `git status`, `git status --short`, `git diff --stat`, and `git diff --name-only`. Other syntactically valid commands with a valid cwd require foreground user confirmation; Scheduled Work and other background execution treat that confirmation requirement as refusal. Users cannot extend the allowlist.

This policy is not an operating-system filesystem or network sandbox. A foreground command explicitly approved by the user can still reference an absolute path outside the Workspace or access the network. The implementation must state this boundary accurately and must not claim that cwd validation confines child-process effects. Commands with an out-of-Workspace cwd, invalid timeout, NUL, or control characters are denied before confirmation.

Strong process isolation is out of scope for the first version because Windows and POSIX require different sandbox mechanisms and a string scanner cannot provide that guarantee. If MyClaw later requires commands to remain confined even after user approval, that version must select and document an OS-level isolation design in a new ADR.
