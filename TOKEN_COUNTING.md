# Token Counting

The agent needs to count tokens to know how many files fit inside the model's
context window before sending a request. Three backends are available,
selected via `tokenizer_backend` in `config.yaml`.

## Which backend to use

| Your setup                                           | Backend              |
| ---------------------------------------------------- | -------------------- |
| Gemma 4 (or any local model with a `tokenizer.json`) | `tokenizers_json`    |
| OpenAI / GPT-4 / GPT-4o                              | `tiktoken`           |
| You don't care / just want it to work                | `estimate` (default) |

## Setup for Gemma 4 — `tokenizers_json` (recommended)

**1. Install the tokenizers library**

```bash
pip install tokenizers
```

This is a small Rust-based package (~10 MB). It is _not_ the same as
`transformers` and does not pull in PyTorch.

**2. Find your tokenizer.json**

Check your model's repo on HF. Download if you haven't yet and put near you local model. It lives next to your GGUF file, for example:

```
/models/gemma-4-E4B-it/
    gemma-4-E4B-it-Q4_K_M.gguf
    tokenizer.json          ← this one
    tokenizer_config.json
    ...
```

Do not copy `tokenizer.json` into this project. Point the config at it
where it already lives.

**3. Add two lines to config.yaml**

```yaml
tokenizer_backend: tokenizers_json
tokenizer_model_path: /models/gemma-4-E4B-it/tokenizer.json
```

Use an absolute path. Tilde expansion (`~/models/…`) also works.

That's it. The file is loaded once on first use and cached in memory for
the rest of the session. No network access, no authentication.

The `tokenizers_json` backend relies on the Hugging Face `tokenizers` library. The rule of thumb is: If the model's repository on Hugging Face contains a **tokenizer.json** file, this approach will work perfectly.

## Setup for OpenAI models — `tiktoken`

```bash
pip install tiktoken
```

```yaml
tokenizer_backend: tiktoken
tokenizer_model: gpt-4o # or gpt-4, gpt-3.5-turbo, etc.
```

---

## Estimate fallback

No configuration or installation required. Used automatically when
`tokenizer_backend` is set to `"estimate"` or is not set at all.

```yaml
tokenizer_backend: estimate # this is also the default if omitted
```

The estimate uses `characters // 3`, which slightly over-counts.
Over-counting is intentional: the agent stays safely inside the context
budget rather than accidentally overflowing it. Expect roughly 10–20 %
wasted context headroom compared to an exact count.

---

## Verifying it works

Run the agent on any project and check the startup line:

```
📂 Context: 42 files | ~18,300 tokens
```

To see which backend is active, temporarily set the log level to DEBUG or
add a one-liner to your entry point:

```python
from your_package.token_counter import count_tokens
from your_package.config import Config

cfg = Config.load()
result = count_tokens("hello world", cfg)
print(result)  # TokenCountResult(count=3, tokenizer_name='tokenizers:gemma-4-E4B-it')
```

The `tokenizer_name` field shows exactly which backend and file were used.
