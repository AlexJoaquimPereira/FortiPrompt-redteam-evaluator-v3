"""
api.py — FortiPrompt HTTP API (FastAPI).

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
POST /evaluate          — evaluate a single session
POST /evaluate/batch    — evaluate multiple sessions at once
GET  /health            — check that the llama.cpp server is reachable

Interactive docs: http://localhost:8000/docs
"""

import logging
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

from evaluator import Evaluator, SessionResult
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="FortiPrompt RedTeam Evaluator",
    description="Evaluates LLM safety using WildGuard via a llama.cpp server.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared Evaluator instance (judge HTTP client is re-used across requests)
_evaluator = Evaluator()


# ── Request / response schemas ────────────────────────────────────────────────

class SessionRequest(BaseModel):
    prompts:       list[str]
    responses:     list[str]
    session_id:    Optional[str]  = None
    attack_method: str            = ""
    target_model:  str            = ""
    benign_turns:  list[int]      = [0]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.prompts) != len(self.responses):
            raise ValueError("prompts and responses must be the same length")
        return self


class BatchRequest(BaseModel):
    sessions: list[SessionRequest]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Check that the llama.cpp server is reachable.
    Returns 200 if healthy, 503 if the llama.cpp server is unreachable.
    """
    try:
        resp = httpx.get(config.LLAMA_URL + "/health", timeout=5)
        llama_ok = resp.status_code == 200
    except Exception:
        llama_ok = False

    if not llama_ok:
        raise HTTPException(
            status_code=503,
            detail=f"llama.cpp server unreachable at {config.LLAMA_URL}",
        )
    return {"status": "ok", "llama_url": config.LLAMA_URL}


@app.post("/evaluate")
def evaluate(req: SessionRequest) -> dict:
    """
    Evaluate a single multi-turn session.

    **Request body**

    ```json
    {
      "prompts":       ["user turn 0", "user turn 1"],
      "responses":     ["model reply 0", "model reply 1"],
      "session_id":    "optional-uuid",
      "attack_method": "GCG",
      "target_model":  "llama-3-8b",
      "benign_turns":  [0]
    }
    ```

    **Response** — SessionResult with per-turn verdicts.

    Verdict values: `BREACH` | `SAFE` | `FAST_REFUSAL` | `ERROR`
    """
    result: SessionResult = _evaluator.run_session(
        prompts=req.prompts,
        responses=req.responses,
        session_id=req.session_id,
        attack_method=req.attack_method,
        target_model=req.target_model,
        benign_turns=req.benign_turns,
    )
    return result.to_dict()


@app.post("/evaluate/batch")
def evaluate_batch(req: BatchRequest) -> dict:
    """
    Evaluate multiple sessions in sequence.

    **Request body**

    ```json
    {
      "sessions": [
        {"prompts": [...], "responses": [...], "attack_method": "GCG"},
        {"prompts": [...], "responses": [...], "attack_method": "PAIR"}
      ]
    }
    ```

    **Response**

    ```json
    {
      "total": 2,
      "results": [ <SessionResult>, <SessionResult> ]
    }
    ```
    """
    sessions_kwargs = [
        {
            "prompts":       s.prompts,
            "responses":     s.responses,
            "session_id":    s.session_id,
            "attack_method": s.attack_method,
            "target_model":  s.target_model,
            "benign_turns":  s.benign_turns,
        }
        for s in req.sessions
    ]
    results = _evaluator.run_batch(sessions_kwargs)
    return {
        "total":   len(results),
        "results": [r.to_dict() for r in results],
    }
