[runtime]
max_tool_result_chars = 50000

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[tools.web]
enabled = true

[tools.shell]
enabled = true

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

# Replace provider_id, model, and model limits with values supported by your provider.
# Remove any purpose-specific route to fall back to default.
[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.chat]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.memory]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.cron]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
