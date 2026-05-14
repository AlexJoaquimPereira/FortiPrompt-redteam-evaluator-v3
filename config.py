"""
config.py — All runtime settings in one place.
Edit this file or set the matching environment variables before starting the server.
"""

import os

# ── llama.cpp server ─────────────────────────────────────────────────────────
# URL of your running llama-server (llama.cpp) or llama-cpp-python server.
# The evaluator posts to  LLAMA_URL + "/completion"
LLAMA_URL: str = os.getenv("LLAMA_URL", "http://localhost:8080")

# Seconds to wait for the llama.cpp server to respond before giving up.
LLAMA_TIMEOUT: int = int(os.getenv("LLAMA_TIMEOUT", "60"))

# ── MongoDB (optional) ───────────────────────────────────────────────────────
# Leave blank to disable persistence entirely.
MONGO_URI: str  = os.getenv("MONGO_URI", "")
MONGO_DB:  str  = os.getenv("MONGO_DB",  "fortiprompt")

# ── FortiPrompt behaviour ────────────────────────────────────────────────────
# Maximum turns to evaluate per session (extra turns are silently dropped).
MAX_TURNS: int = int(os.getenv("MAX_TURNS", "20"))
