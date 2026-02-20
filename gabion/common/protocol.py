from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


MessageType = Literal[
    "register",
    "registered",
    "heartbeat",
    "list_jobs",
    "job_list",
    "join_job",
    "job_joined",
    "job_rejected",
    "artifact_required",
    "round_start",
    "round_result",
    "round_summary",
    "error",
]


class Message(TypedDict):
    type: MessageType
    payload: Dict[str, Any]


class RoundResultPayload(TypedDict):
    worker_id: str
    job_id: str
    round_id: int
    sample_count: int
    weights: List[float]
    metrics: Dict[str, float]


def make_message(msg_type: MessageType, payload: Dict[str, Any]) -> Message:
    return {"type": msg_type, "payload": payload}
