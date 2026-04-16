# ◉ CodeOrbit

![Python Version](https://img.shields.io/badge/python-3.14-blue)
![Backend](https://img.shields.io/badge/backend-Local--LLM-orange)
![Interface](https://img.shields.io/badge/UI-Rich-green)
![Type](https://img.shields.io/badge/type-CLI-informational)
![License](https://img.shields.io/badge/license-MIT-success)

A local, agentic code editor compatible with any OpenAI-compatible local LLM provider ([llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai)). Point it at any directory, give it a prompt, and it reads your codebase, plans changes, shows you a diff, and applies them.

No cloud. No Docker. No API keys.

## How it works

```
your prompt
    │
    ▼
build context          ← walks your directory, respects .gitignore-style patterns
    │                    fits files within your model's context window
    ▼
call LLM provider    ← OpenAI-compatible REST API (llama.cpp, Ollama, etc.)
    │                    model returns JSON: { summary, changes[] }
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

## Setup

```bash
# 1. Clone
git clone https://github.com/yourname/code-orbit
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
  --n-gpu-layers 99

# 5. Copy and edit config
cp config.yaml config.local.yaml   # optional — config.yaml works as-is

# 6. Using other local providers (Optional)
Code Orbit works with any OpenAI-compatible local server.

**Ollama**
1. Start Ollama and pull your model (e.g., `ollama pull qwen2.5-coder:32b`).
2. Update `config.yaml` or use the `--profile ollama` flag.
3. Set `api_base: http://localhost:11434/v1`.

**LM Studio**
1. Open LM Studio and start the "Local Server".
2. Set `api_base: http://localhost:1234/v1` in your config.
3. Use the model identifier provided by LM Studio in the `model` field.
```

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

Tokens are counted using `tokenizer_backend: estimate`. If you want to learn more on how to use tokenizers and improve tokens counting see TOKEN_COUNTING.md.

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
interactive: true # show diffs, ask before applying
auto_commit: false # git commit after applying
```

For personal overrides without affecting git, use `config.local.yaml` (already in `.gitignore`).

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
| `gpt-oss-20b`     | 21B             | 128K    | High-performance open model from Oracle/OpenAI for STEM and coding.                       |
| `phi-4`           | 14B             | 64K     | Dense reasoning powerhouse; excels at complex math and scientific logic.                  |
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
  --n-gpu-layers 99 \         # offload all layers to GPU
  --threads 8 \               # CPU threads for non-GPU layers
  --batch-size 512            # prompt processing batch size
```

## Project structure

```
code-orbit/
├── main.py              # CLI entry point
├── agent/
│   ├── config.py        # Config dataclass, YAML loading
│   ├── context.py       # Directory walker, context builder
│   ├── llm.py           # llama.cpp API client
│   └── patcher.py       # Diff preview, file writer, git commit
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
- [ ] Two-pass planning and execution workflow
- [ ] `.agentignore` support for project-specific exclusions
- [ ] Web UI with browser-based workflow

## License

MIT © 2026 Olha Stefanishyna
