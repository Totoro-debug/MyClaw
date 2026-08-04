# MyClaw

MyClaw is a host-neutral, local-first Personal Agent runtime for Python 3.12 or
newer. It has no platform gate: Windows selects native Windows adapters and other
hosts attempt the POSIX adapters when an operation needs them.

## Install

Create a virtual environment, install the project, and run the console entry point.
On Windows x64, the currently validated environment:

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\myclaw.exe
```

For a release wheel, replace `.` in the install line with the wheel path, for example
`.venv\Scripts\python.exe -m pip install dist\myclaw-0.1.0-py3-none-any.whl`.
The remaining examples use `myclaw` for readability; activate the virtual environment
first or use the full console path shown above.

The same `py3-none-any` wheel contains the Windows and POSIX host adapters. macOS
Intel and Apple Silicon are intended compatibility targets but remain unverified;
fake-adapter coverage is not native macOS validation. Linux and other POSIX hosts may
attempt the POSIX adapter, but this release makes no formal support claim for them.

The first start creates `~/.myclaw/config.toml` and the base
Agent Home, prints `config_missing`, and exits with status 2. Edit that file before
starting the REPL. `myclaw config` prints the current file with API keys redacted,
including when the configuration is invalid.

## Configure

The generated file contains runtime, memory, Tool, and Provider defaults plus editable
scaffolds for the `default`, `chat`, `memory`, and `cron` Model Routes. It contains one
`openai-compatible` Provider scaffold. Complete that Provider, then replace the
generated route values using a supported model. A minimal working configuration can
keep only the `default` route:

```toml
[models.providers.openai-local]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "replace-with-a-dedicated-key"
models = ["replace-with-a-model-id"]

[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
```

Providers use either `anthropic` or `openai-compatible` protocol. The `chat` route is
used for conversations and Session titles, `memory` for summaries and Memory Tasks,
and `cron` for Scheduled Work. Remove any purpose-specific route table to fall back to
`default`; Provider model IDs must also appear in that Provider's `models` array. The
schema is strict: unknown tables or fields are configuration errors.

`api_key` values are plaintext at rest in `config.toml`. MyClaw redacts keys from
configuration views and user-visible errors, but v0.1 has no environment-variable
reference or operating-system keychain integration. Protect the host account and
Agent Home permissions, and use a dedicated key with the minimum required access.

## Run

Start `myclaw` from the Workspace the agent should operate in. The current directory
defines the Workspace boundary and the Session group. The REPL supports normal chat
plus these built-in management commands:

- `/config`: show redacted configuration.
- `/status`: show runtime, model, token, and Session status.
- `/resume`: select a Session from the current Workspace.
- `/memory`: show the latest Long-term Memory file.
- `/dream`: process pending Conversation Summaries now.

Use `exit` or `quit` to shut down. Ctrl+C cancels the active foreground turn while
the REPL remains available. File changes and non-automatic Shell commands require
foreground confirmation; background work refuses operations that would require a
prompt.

## Persistent State

MyClaw keeps global User Configuration in the current account's fixed Agent Home at
`~/.myclaw/`. Legacy Agent Home Runtime Log files remain untouched: upgrades never
read, move, delete, truncate, or update existing `run.log.0`, `run.log.1`,
`run.log.cursor`, or `run.log.lock` files.

```text
~/.myclaw/
  config.toml
```

Every startup directory is an independent Workspace. Its non-global state lives in
the reserved `.myclaw` directory beneath that Workspace:

```text
<workspace>/.myclaw/
  .gitignore
  scheduled-work.json
  memory/
    memory.md
    summary.jsonl
    .cursor
  sessions/
    <session_id>.jsonl
    artifacts/<session_id>/<encoded_tool_call_id>.txt
  logs/
    <session_id>.log
```

Valid REPL startup creates only the Workspace State root, `.gitignore`, `memory/`,
`sessions/`, and `memory/memory.md`; `logs/` is created lazily by an explicit Session
context when a WARNING or ERROR is emitted. Summary, Summary Cursor, Scheduled Work,
Session, and Artifact files remain on demand. Old non-global Agent Home data is ignored
and is never migrated or deleted. Back up each Workspace State directory with its
Workspace; do not edit active Session, Summary, Summary Cursor, or Scheduled Work files.

Each foreground Runtime has one active Conversation Session. Its in-memory messages
are ordinary JSON-compatible dictionaries and its metadata is a JSON-compatible
dictionary. A nonempty Session is written as a complete compact JSONL snapshot: the
first line has exactly `session_id`, `created_at`, `updated_at`, `last_consolidated`,
and `metadata`, and each later line is a user, assistant, or tool message with a
local-time `timestamp`. Session state is changed in memory during a turn; after terminal
work `persist()` schedules an ordered atomic replacement, while `close()` makes at most
three bounded synchronous save attempts. Ordinary save failures are silent and provide
no acknowledgement or failure log. Old Session schemas are unsupported, with no
migration or version dispatch. A late generated title may wait for a later turn or
shutdown save, and a crash can leave Conversation Summary and `last_consolidated`
temporarily divergent.

Session Logs use Loguru with a WARNING threshold, an unbounded queue, exact 10 MiB
rotation, and per-Session retention of at most one historical file. Same-Session
concurrency is unsupported, both within one process and across processes. Normal
context exit performs an infinite drain. There is no per-record fsync, so crashes,
power loss, or forced termination can lose recent records. No active redaction and
no control escaping are performed: exception messages, credentials, newlines, and
other control characters supplied to a log call may be stored verbatim. Retention is
per Session only, so total Workspace log usage is unbounded across Sessions.

## Troubleshooting

- `config_missing`: edit the generated `~/.myclaw/config.toml`, define a usable
  `default` route, then run `myclaw` again.
- `config_parse_error` or `config_invalid`: run `myclaw config` for a redacted view,
  then correct TOML syntax, unknown fields, Provider details, or route values.
- `route_unavailable`: ensure `models.routes.default` references a configured
  Provider and a model listed by that Provider.
- Provider authentication, timeout, or connection failures: verify the dedicated
  key, base URL, model availability, account policy, and network path. MyClaw does
  not automatically retry permanent authentication or invalid-request failures.
- `memory_context_too_large`: reduce `<workspace>/.myclaw/memory/memory.md` or use a route with a larger
  context window. Long-term Memory is injected in full and has no automatic size cap.
- Corrupt JSON/JSONL or Session load errors: stop all MyClaw processes, back up the
  affected Workspace State and Session Logs, and restore a known-good file. Loads fail
  closed instead of discarding complete but invalid records. Ordinary background
  Session save failures are silent and do not produce a troubleshooting error.

## Known Limits

- Multiple REPL processes do not coordinate Session writes, Session Logs, or background
  schedules. In particular, same-Session concurrency is unsupported.
- A failed background Session snapshot can lose the latest in-memory turn after an
  abnormal process exit; Session persistence has no ordinary acknowledgement or failure
  logging. Conversation Summary and `last_consolidated` can diverge after a crash.
- Approved Shell commands are not an operating-system sandbox and can affect more
  than the Workspace according to the user's OS permissions.
- Long-term Memory has no automatic size cap, and Tool Artifacts have no automatic
  cleanup policy.
- v0.1 has no daemon, HTTP/IPC service, MCP support, subagent runtime, profiles,
  cross-process coordination for Workspace state or Session Logs, keychain integration, or
  environment-variable API keys.

The full acceptance record and additional security limits are in
[docs/release-readiness.md](docs/release-readiness.md).

## License

MyClaw is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for the full terms.

## Development

After the project dependencies are available in the local Python environment, all
verification commands run without network access:

```text
python -m pip install --no-index --no-deps --no-build-isolation -e .
pytest
ruff check .
ruff format --check .
mypy myclaw tests
python -m build --wheel
```

The automated tests use scripted boundary fakes and temporary filesystem paths. They do
not call model providers or other external services.
