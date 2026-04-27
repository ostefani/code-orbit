# ◉ CodeOrbit

![Python Version](https://img.shields.io/badge/python-3.14-blue)
![Backend](https://img.shields.io/badge/backend-Local--LLM-orange)
![Interface](https://img.shields.io/badge/UI-Rich-green)
![Type](https://img.shields.io/badge/type-CLI-informational)
![License](https://img.shields.io/badge/license-MIT-success)

A local, agentic code editor compatible with any OpenAI-compatible local LLM provider ([llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai)). Supports Point it at any directory, give it a prompt, and it reads your codebase, plans changes, shows you a diff, and applies them.

No cloud. No Docker. No API keys.

## Demo

https://github.com/user-attachments/assets/2481e3f3-1b40-4340-bfa0-9ab1d45be341

## How it works

```
your prompt
    │
    ▼
build context          ← walks your directory, respects .gitignore-style patterns
    │                    fits files within your model's context window
    ▼
chat provider         ← provider-selected chat adapter powers planning/editing
    │
    ▼
architect plan        ← high-level JSON plan with files, goals, and reasoning
    │
    ▼
review / edit plan     ← user can edit .code-orbit/plan.json before approval
    │
    ▼
coder response        ← exact file replacements from the approved plan
    ▼
preview diff           ← colored unified diff per file
    │
    ▼
confirm → apply        ← writes files to disk, optional git commit
```

## Requirements

- Python 3.14+ (relies on `pathlib.Path.walk`)
- A local LLM server (llama.cpp, Ollama, or LM Studio)
- A GGUF model (see [recommended models](#recommended-models))

## Downloads

GitHub Releases publish the Python package artifacts (`.whl` and source tarball).
Install from a release or from source using the setup steps below.

## Setup

```bash
# 1. Clone
git clone https://github.com/ostefani/code-orbit
cd code-orbit

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start llama.cpp server (in a separate terminal)
llama-server \
  --model /path/to/your-model.gguf \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers -1

# Get model name for configs
curl http://localhost:<PORT>/v1/models | jq '.data[].id'

# 5. Copy and fill config with your data
cp config.yaml config.local.yaml

# 6. Configure chat and embeddings (Optional)
# Chat goes through a provider adapter layer selected from config. The
# current implementation ships with an OpenAI/OpenAI-compatible chat adapter.

chat_provider: openai
chat_api_base: http://localhost:8081/v1
chat_api_key: dummy
chat_model: local
chat_context_window: 16384
chat_streaming: true
chat_probe_on_startup: false

embedding_provider: openai
embedding_api_base: http://localhost:8081/v1
embedding_api_key: dummy
embedding_model: nomic-embed-text
embedding_probe_on_startup: false
embedding_provider_options:
  timeout: 30
```

`chat_*` settings control the provider used for planning and code generation.
`structured_*` settings control JSON-structured LLM output behavior.
`embedding_*` settings control semantic retrieval. Provider-specific SDK
settings belong in the matching `*_provider_options` mapping.

Chat lifecycle at a glance:

- `build_chat_adapter()` constructs an unvalidated adapter.
- `validate_chat_adapter()` runs local checks only.
- `probe_chat_adapter()` performs an optional live readiness check for
  probing-capable adapters.
- `create_chat_adapter()` is the startup convenience path that combines build,
  validation, and optional probe.

Set `chat_probe_on_startup: true` if you want Code Orbit to make one live chat
provider readiness check at startup to verify credentials and reachability.
For OpenAI-compatible providers this is a live `models.list()` request, so
leave it off to keep startup cheap.
Set `embedding_probe_on_startup: true` if you want Code Orbit to make one live
embedding request at startup to verify credentials and reachability. Leave it
off to keep startup cheap and let the first semantic operation hit the backend.

### Using other local providers (Optional)

Code Orbit still works with any OpenAI-compatible local server.

**Ollama**

1. Start Ollama and pull your model.
2. Update `config.yaml` or use the `--profile ollama` flag.
3. Set `chat_api_base: http://localhost:11434/v1` and `chat_model` to the model name you pulled.

**LM Studio**

1. Open LM Studio and start the "Local Server".
2. Set `chat_api_base: http://localhost:1234/v1` in your config.
3. Use the model identifier provided by LM Studio in the `chat_model` field.

## Usage

```bash
# Basic — edit current directory
python main.py --prompt "Add type hints to all functions"

# Target a specific project
python main.py --dir ~/projects/my-api --prompt "Add request logging middleware"

# Skip confirmation prompt
python main.py --dir . --prompt "Fix all f-string formatting issues" --no-interactive

# Auto git commit after applying
python main.py --dir . --prompt "Rename userId to user_id everywhere" --auto-commit

# Debug: see what files would be included in context
python main.py --dir . --prompt "" --tree

# Use a different config file
python main.py --dir . --prompt "..." --config config.local.yaml
```

When you edit the plan, Code Orbit reads `EDITOR` as a command line rather than a shell string, so values like `vim -u NONE` or `code --wait` work. The launcher is currently tuned for Unix-like environments; if Windows support becomes a goal, the editor parsing strategy will need a small portability review because `shlex.split()` uses POSIX rules.

Tokens are counted using `tokenizer_backend: estimate`. If you want to learn more on how to use tokenizers and improve tokens counting see TOKEN_COUNTING.md.

## Public API

Code Orbit now exposes a small Python API for external runners and other Python consumers.

```python
from pathlib import Path

from api import AgentRunRequest, AgentRunStatus
from workflow.core import run_workflow_core

request = AgentRunRequest(
    target_dir=Path("~/projects/my-app").expanduser(),
    prompt="Add request logging to every endpoint",
    auto_commit=False,
    allow_delete=False,
)
```

`AgentRunRequest` is the public input model. It carries the run identifier, target directory, prompt, and the per-run policy flags that callers may override.

`run_workflow_core()` is the reusable workflow entrypoint. It accepts an `AgentRunRequest`, a loaded `Config`, and an `EventBus`, then returns an `AgentRunResult`.

`AgentRunResult` reports the final status, summary or answer, optional error, affected files, and timestamps. `run_id` is available as a property copied from the original request.

`AgentRunStatus` describes the lifecycle returned by the API:

- `queued`
- `running`
- `completed`
- `answered`
- `failed`
- `cancelled`

The CLI wrapper `workflow.run_workflow()` remains the terminal-facing entrypoint. It loads config, publishes lifecycle events, and then delegates the actual run to `run_workflow_core()`.

For event-driven consumers, `agent.events.AsyncEventQueue` can be subscribed to the event bus and consumed asynchronously to stream progress without blocking the workflow.

### All options

| Flag               | Default       | Description                |
| ------------------ | ------------- | -------------------------- |
| `--dir`, `-d`      | `.`           | Directory to edit          |
| `--prompt`, `-p`   | _(required)_  | What change to make        |
| `--config`, `-c`   | `config.yaml` | Config file path           |
| `--no-interactive` | false         | Apply without confirmation |
| `--auto-commit`    | false         | Git commit after applying  |
| `--tree`           | false         | Print file tree and exit   |
| `--profile`        | gemma-e4b     | Provide model name         |

## Configuration

Edit `config.yaml`:

```yaml
api_base: 'http://localhost:8080/v1' # llama.cpp server
max_context_tokens: 16384 # match your model's context size
max_response_tokens: 4096 # reserve room for the model's reply
structured_llm_temperature: 0.2 # JSON output tuning knob
structured_llm_retries: 1 # retry malformed JSON and transient provider errors
structured_llm_retry_delay_seconds: 1.0 # base delay for rate-limit retries
chat_provider: openai # chat provider adapter
chat_api_base: 'http://localhost:8080/v1'
chat_api_key: 'dummy'
chat_model: 'local'
chat_context_window: 16384
chat_streaming: true
chat_probe_on_startup: false
interactive: true # show diffs, ask before applying
auto_commit: false # git commit after applying
embedding_provider: openai # embeddings provider adapter
embedding_api_base: 'http://localhost:8081/v1'
embedding_api_key: '' # leave empty to reuse api_key
embedding_model: 'nomic-embed-text'
embedding_probe_on_startup: false # set true to probe the backend once at startup
embedding_provider_options: {} # provider-specific SDK kwargs
chat_provider_options: {} # provider-specific SDK kwargs
```

For personal overrides without affecting git, use `config.local.yaml` (already in `.gitignore`).

Code Orbit also stores runtime artifacts under `.code-orbit/`, including the embeddings cache and prompt history. That directory is ignored in this repo by default, so you should not commit those generated files.

## Testing

Run the tests:

```bash
python3 -m pytest tests/
python3 main.py
```

## Recommended models

These models are optimized for the agentic workflows, long context, and structured JSON output required by Code Orbit.

| Model             | Size            | Context | Notes                                                                                     |
| :---------------- | :-------------- | :------ | :---------------------------------------------------------------------------------------- |
| `qwen3-coder-30b` | 30B (3B active) | 256K    | **Best overall for coding.** State-of-the-art MoE with 128 experts for agentic tool use.  |
| `qwen3.6-plus`    | (Varies)        | 1M      | Specialized **reasoning** model built for step-by-step execution and long-form workflows. |
| `gemma-4-26b-a4b` | 26B (4B active) | 256K    | Extreme MoE efficiency; delivers 27B-class intelligence at 4B compute cost.               |
| `gemma-4-e4b`     | 4B              | 128K    | Fast, mobile-first model with native multimodal (audio/vision) capabilities.              |

---

### Key Updates for 2026 Models:

- **Agentic Capabilities**: Models like **Qwen3-Coder** and **Gemma 4** now feature native support for agentic workflows and function-calling, significantly reducing JSON hallucination.
- **MoE Architecture**: The transition to high-expert-count Mixture-of-Experts (e.g., Gemma 4's 128 experts) allows you to run high-intelligence models with the latency of a much smaller model.
- **Expanded Context**: Standard context windows have moved from 32K to **128K–256K**, with reasoning models like Qwen3.6 supporting up to **1M tokens**, allowing the agent to ingest entire repositories at once.

Download AI model from [Hugging Face](https://huggingface.co), convert to GGUF.

### llama-server flags that matter

```bash
llama-server \
  --model your-model.gguf \
  --port 8080 \
  --ctx-size 32768 \          # must match config.yaml max_context_tokens
  --n-gpu-layers -1 \         # offload all layers to GPU
  --threads 8 \               # CPU threads for non-GPU layers
  --batch-size 512            # prompt processing batch size
  --embeddings \
  --pooling mean
```

## Project structure

```
code-orbit/
├── main.py              # CLI wrapper, history, prompt handling
├── api/                 # Public request / result / status models
├── agent/
│   ├── config.py        # Config dataclass, YAML loading
│   ├── context.py       # Directory walker, context builder
│   ├── chat/            # Provider-agnostic chat adapters and factory
│   ├── llm.py           # Structured planning/coder wrappers on chat adapters
│   └── patcher.py       # Diff preview, file writer, git commit
├── workflow/
│   ├── __init__.py      # workflow entrypoint and orchestration
│   ├── _state.py        # workflow state and runtime dataclass
│   ├── context.py       # context-building stage
│   ├── planning.py      # plan drafting and review stage
│   ├── editing.py       # plan editing stage
│   ├── execution.py     # coder execution and validation stage
│   └── output.py        # preview / apply / commit stages
├── config.yaml          # Default config (commit this)
├── config.local.yaml    # Personal overrides (gitignored)
├── requirements.txt
└── .vscode/
    ├── settings.json
    ├── launch.json      # Debug configs with prompt inputs
    └── extensions.json
```

## Roadmap

- [x] Direct codebase inclusion for context building
- [x] JSON-only change output with patch application
- [x] Tests for context building and patch application
- [ ] Embeddings-based retrieval (RAG) for semantic file selection
- [ ] Retry on invalid model output with stricter JSON recovery
- [ ] File watch mode for optional automatic re-run on code changes
- [x] Two-pass planning and execution workflow
- [ ] `.agentignore` support for project-specific exclusions
- [ ] Web UI with browser-based workflow

## Contributing

Please see CONTRIBUTING.md for contribution and PR guidelines.

## License

MIT © 2026 Olha Stefanishyna
