"""
evaluator.py — Multi-turn session evaluator.

Takes two plain lists (prompts, responses) and returns a SessionResult.
Optionally writes to MongoDB if MONGO_URI is configured.

Typical use
-----------
    from evaluator import Evaluator

    ev = Evaluator()
    result = ev.run_session(
        prompts=["seed", "attack payload"],
        responses=["OK", "Sure, here's how…"],
        session_id="my-session",          # optional
        attack_method="DirectRequest",
        target_model="llama-3-8b",
        benign_turns=[0],                 # turn 0 is a benign probe
    )
    print(result.status)        # "BREACH" or "EXHAUSTED"
    print(result.breach_turn)   # index of first breach, or None
    print(result.to_dict())     # JSON-ready dict

Batch use
---------
    results = ev.run_batch([
        {"prompts": [...], "responses": [...], "attack_method": "GCG"},
        {"prompts": [...], "responses": [...], "attack_method": "PAIR"},
    ])
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from judge import Judge, JudgeResult, Verdict
import config

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn_index:    int
    prompt:        str
    response:      str
    verdict:       str
    is_breach:     bool
    is_benign:     bool
    latency_ms:    float
    intent_harm:   Optional[bool]
    response_harm: Optional[bool]
    refusal:       Optional[bool]
    raw_output:    str
    error:         Optional[str]

    def to_dict(self) -> dict:
        return {
            "turn_index":    self.turn_index,
            "verdict":       self.verdict,
            "is_breach":     self.is_breach,
            "is_benign":     self.is_benign,
            "latency_ms":    round(self.latency_ms, 2),
            "labels": {
                "intent_harm":   self.intent_harm,
                "response_harm": self.response_harm,
                "refusal":       self.refusal,
            },
            "raw_output": self.raw_output,
            "error":      self.error,
        }


@dataclass
class SessionResult:
    session_id:    str
    status:        str                        # "BREACH" | "EXHAUSTED"
    breach_turn:   Optional[int]
    attack_method: str
    target_model:  str
    turns:         list[TurnResult] = field(default_factory=list)

    @property
    def breached(self) -> bool:
        return self.status == "BREACH"

    @property
    def ttb(self) -> Optional[int]:
        """Turns-to-Breach: index of first breach, or None."""
        return self.breach_turn

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "status":        self.status,
            "breach_turn":   self.breach_turn,
            "ttb":           self.ttb,
            "breached":      self.breached,
            "attack_method": self.attack_method,
            "target_model":  self.target_model,
            "turns":         [t.to_dict() for t in self.turns],
        }


# ── Optional MongoDB helper ───────────────────────────────────────────────────

def _get_db():
    """Return the MongoDB database object, or None if not configured."""
    if not config.MONGO_URI:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5_000)
        client.admin.command("ping")
        return client[config.MONGO_DB]
    except Exception as exc:
        logger.warning("MongoDB unavailable (%s) — running without persistence.", exc)
        return None


def _save_to_mongo(db, session: SessionResult) -> None:
    """Write/update one session document and its turns to MongoDB."""
    if db is None:
        return
    try:
        db["sessions"].replace_one(
            {"session_id": session.session_id},
            session.to_dict(),
            upsert=True,
        )
        for turn in session.turns:
            db["turns"].replace_one(
                {"session_id": session.session_id, "turn_index": turn.turn_index},
                {"session_id": session.session_id, **turn.to_dict()},
                upsert=True,
            )
    except Exception as exc:
        logger.warning("MongoDB write failed: %s", exc)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs multi-turn evaluation sessions using the Judge.

    Parameters
    ----------
    judge     : A Judge instance. Created with defaults if not provided.
    persist   : If True, save results to MongoDB (requires MONGO_URI in config).
    max_turns : Cap on the number of turns to evaluate per session.
    """

    def __init__(
        self,
        judge:     Optional[Judge] = None,
        persist:   bool = True,
        max_turns: int  = config.MAX_TURNS,
    ):
        self._judge     = judge or Judge()
        self._max_turns = max_turns
        self._db        = _get_db() if persist else None

    def run_session(
        self,
        prompts:       list[str],
        responses:     list[str],
        session_id:    Optional[str] = None,
        attack_method: str = "",
        target_model:  str = "",
        benign_turns:  list[int] = None,
    ) -> SessionResult:
        """
        Evaluate a session expressed as two parallel lists.

        Parameters
        ----------
        prompts       : Ordered attacker / user inputs.
        responses     : Ordered target model outputs (same length as prompts).
        session_id    : Optional UUID. Auto-generated if not supplied.
        attack_method : Label for the attack type (stored in DB, used for metrics).
        target_model  : Label for the model under test.
        benign_turns  : 0-based indices of benign probe turns. Default: [0].
                        Benign turns are evaluated but flagged is_benign=True.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")

        sid          = session_id or str(uuid.uuid4())
        benign_set   = set(benign_turns) if benign_turns is not None else {0}
        n            = min(len(prompts), self._max_turns)

        turns:        list[TurnResult] = []
        breach_turn:  Optional[int]    = None

        for i in range(n):
            is_benign = i in benign_set
            jr: JudgeResult = self._judge.evaluate(prompts[i], responses[i])

            tr = TurnResult(
                turn_index=i,
                prompt=prompts[i],
                response=responses[i],
                verdict=jr.verdict.value,
                is_breach=jr.is_breach,
                is_benign=is_benign,
                latency_ms=jr.latency_ms,
                intent_harm=jr.intent_harm,
                response_harm=jr.response_harm,
                refusal=jr.refusal,
                raw_output=jr.raw_output,
                error=jr.error,
            )
            turns.append(tr)

            logger.info(
                "[%s] turn %d → %s  (%.0f ms)  I=%s H=%s R=%s",
                sid, i, jr.verdict.value, jr.latency_ms,
                jr.intent_harm, jr.response_harm, jr.refusal,
            )

            if jr.is_breach and breach_turn is None:
                breach_turn = i
                logger.info("[%s] *** BREACH at turn %d ***", sid, i)

        status = "BREACH" if breach_turn is not None else "EXHAUSTED"
        result = SessionResult(
            session_id=sid,
            status=status,
            breach_turn=breach_turn,
            attack_method=attack_method,
            target_model=target_model,
            turns=turns,
        )

        _save_to_mongo(self._db, result)
        return result

    def run_batch(self, sessions: list[dict]) -> list[SessionResult]:
        """
        Evaluate a list of sessions.

        Each dict in ``sessions`` is passed as **kwargs to run_session().

        Example
        -------
            ev.run_batch([
                {"prompts": [...], "responses": [...], "attack_method": "GCG"},
                {"prompts": [...], "responses": [...], "attack_method": "PAIR"},
            ])
        """
        results = []
        for i, kwargs in enumerate(sessions):
            logger.info("Batch: session %d / %d", i + 1, len(sessions))
            results.append(self.run_session(**kwargs))
        return results

    def close(self):
        self._judge.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
