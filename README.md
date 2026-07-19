# MyClaw

MyClaw is a local-first Personal Agent runtime for Python 3.12 and newer.

## Install

Create a virtual environment, install the project, and run the console entry point
with the commands for the current platform.

Windows:

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\myclaw.exe
```

POSIX:

```text
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/myclaw
```

For a release wheel, replace `.` in the install line with the wheel path, for example
`.venv\Scripts\python.exe -m pip install dist\myclaw-0.1.0-py3-none-any.whl` on
Windows or `.venv/bin/python -m pip install dist/myclaw-0.1.0-py3-none-any.whl` on
POSIX. The remaining examples use `myclaw` for readability; activate the virtual
environment first or use the full console path shown above.

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

## Agent Home

MyClaw uses one fixed, user-owned Agent Home at `~/.myclaw/`:

```text
~/.myclaw/
  config.toml
  scheduled-work.json
  memory/
    memory.md
    summary.jsonl
    .cursor
    pending-consolidations/
  sessions/
    <workspace_slug>/
      <session_id>.jsonl
      artifacts/<session_id>/<encoded_tool_call_id>.txt
```

Only `memory/`, `sessions/`, and the initial `memory/memory.md` are guaranteed after
first start. Summary, cursor, Scheduled Work, Workspace Session, journal, and Artifact
files are created on demand. Back up the whole directory together; do not edit
Session, summary, cursor, or Scheduled Work files while MyClaw is running.

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
- `memory_context_too_large`: reduce `memory/memory.md` or use a route with a larger
  context window. Long-term Memory is injected in full and has no automatic size cap.
- Persistence errors or corrupt JSON/JSONL: stop all MyClaw processes, back up Agent
  Home, and restore a known-good file. The runtime fails closed instead of silently
  discarding complete but invalid records.

## Known Limits

- Multiple REPL processes do not coordinate Session writes or background schedules.
- Approved Shell commands are not an operating-system sandbox and can affect more
  than the Workspace according to the user's OS permissions.
- Long-term Memory has no automatic size cap, and Tool Artifacts have no automatic
  cleanup policy.
- v0.1 has no daemon, HTTP/IPC service, MCP support, subagent runtime, profiles,
  cross-process locking, keychain integration, or environment-variable API keys.

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
python -m build
```

The automated tests use scripted boundary fakes and temporary filesystem paths. They do
not call model providers or other external services.
