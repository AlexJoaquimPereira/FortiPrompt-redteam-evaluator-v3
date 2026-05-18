"""
config.py — All runtime settings in one place.
Edit this file or set the matching environment variables before starting the server.

JUDGE_BACKEND controls which backend the Judge uses:

  "llamacpp"      — Original mode. Talks to a running llama.cpp /completion server.
                    Requires LLAMA_URL to point at it.

  "transformers"  — Loads the HuggingFace model in-process via the transformers
                    library (no GGUF, no llama.cpp needed). Slower to start but
                    zero external dependencies.  Configure with HF_MODEL_ID.

  "remote"        — Delegates all inference to a remote FortiPrompt Judge Server
                    (judge_server.py running on another machine). Configure with
                    JUDGE_SERVER_URL.
"""

import os

# ── Judge backend selection ───────────────────────────────────────────────────
# One of: "llamacpp" | "transformers" | "remote"
JUDGE_BACKEND: str = os.getenv("JUDGE_BACKEND", "llamacpp")

# ── llama.cpp backend ─────────────────────────────────────────────────────────
# URL of your running llama-server (llama.cpp) or llama-cpp-python server.
# The evaluator posts to  LLAMA_URL + "/completion"
LLAMA_URL:     str = os.getenv("LLAMA_URL",     "http://localhost:8080")
LLAMA_TIMEOUT: int = int(os.getenv("LLAMA_TIMEOUT", "60"))

# ── Transformers backend ──────────────────────────────────────────────────────
# HuggingFace model ID (or local path) to load with AutoModelForCausalLM.
# Default is the Wildguard-Qwen3-4b model.
HF_MODEL_ID: str = os.getenv(
    "HF_MODEL_ID",
    "Kotovskiy/Wildguard-Qwen3-4b",
)

# torch dtype to load the model in.  "auto" lets transformers decide.
# Other sensible values: "float16", "bfloat16", "float32"
HF_TORCH_DTYPE: str = os.getenv("HF_TORCH_DTYPE", "auto")

# Device map passed to from_pretrained.  "auto" spreads across all GPUs/CPU.
# Set to "cpu" to force CPU-only inference.
HF_DEVICE_MAP: str = os.getenv("HF_DEVICE_MAP", "auto")

# Maximum new tokens to generate from the judge model.
HF_MAX_NEW_TOKENS: int = int(os.getenv("HF_MAX_NEW_TOKENS", "64"))

# ── Remote judge backend ──────────────────────────────────────────────────────
# Full base-URL of a machine running judge_server.py.
# e.g. "http://192.168.1.42:8081"  or  "http://my-judge-host:8081"
JUDGE_SERVER_URL:     str = os.getenv("JUDGE_SERVER_URL",     "http://localhost:8081")
JUDGE_SERVER_TIMEOUT: int = int(os.getenv("JUDGE_SERVER_TIMEOUT", "60"))

# ── MongoDB (optional) ────────────────────────────────────────────────────────
# Leave blank to disable persistence entirely.
MONGO_URI: str = os.getenv("MONGO_URI", "")
MONGO_DB:  str = os.getenv("MONGO_DB",  "fortiprompt")

# ── FortiPrompt behaviour ─────────────────────────────────────────────────────
# Maximum turns to evaluate per session (extra turns are silently dropped).
MAX_TURNS: int = int(os.getenv("MAX_TURNS", "20"))
