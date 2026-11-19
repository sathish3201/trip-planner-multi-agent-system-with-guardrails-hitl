import os
import time
from typing import Any

from langgraph.config import get_stream_writer


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Adjust rates per 1M tokens via environment variables or defaults
INPUT_COST_PER_1M_USD = _env_float("INPUT_COST_PER_1M_USD", 0.15)
OUTPUT_COST_PER_1M_USD = _env_float("OUTPUT_COST_PER_1M_USD", 0.60)
USD_TO_INR = _env_float("USD_TO_INR", 84.0)

_ERROR_PHRASES = (
    "unavailable",
    "failed",
    "error",
    "traceback",
    "exception",
    "could not be generated",
    "insufficient token budget",
)


def merge_agent_metrics(
    left: dict[str, dict[str, Any]] | None,
    right: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Reducer for LangGraph state so parallel agent branches merge cleanly."""
    merged: dict[str, dict[str, Any]] = dict(left or {})
    for agent_name, stats in (right or {}).items():
        if isinstance(stats, dict):
            merged[agent_name] = dict(stats)
    return merged


def emit_telemetry_event(
    event_type: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    try:
        writer = get_stream_writer()
        writer({
            "event": event_type,
            "message": message,
            "data": data or {},
        })
    except Exception:
        pass


def extract_token_breakdown(
    ai_message: Any,
    before_usage: int,
    after_usage: int,
) -> dict[str, int]:
    meta = getattr(ai_message, "usage_metadata", None) or {}
    if isinstance(meta, dict) and meta.get("total_tokens") is not None:
        in_tok = int(meta.get("input_tokens", 0) or 0)
        out_tok = int(meta.get("output_tokens", 0) or 0)
        tot_tok = int(meta.get("total_tokens", in_tok + out_tok) or 0)
        return {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
        }

    delta = max(0, int(after_usage or 0) - int(before_usage or 0))
    return {
        "input_tokens": 0,
        "output_tokens": delta,
        "total_tokens": delta,
    }


def compute_cost_usd(input_tokens: int, output_tokens: int) -> float:
    in_cost = (max(0, int(input_tokens)) / 1_000_000) * INPUT_COST_PER_1M_USD
    out_cost = (max(0, int(output_tokens)) / 1_000_000) * OUTPUT_COST_PER_1M_USD
    return round(in_cost + out_cost, 6)


def score_text_accuracy(
    text: str,
    *,
    error_prefixes: tuple[str, ...] = (),
) -> float:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0

    lowered = cleaned.lower()
    if any(lowered.startswith(prefix.lower()) for prefix in error_prefixes):
        return 25.0
    if any(phrase in lowered for phrase in _ERROR_PHRASES):
        return 40.0

    return 100.0 if len(cleaned) >= 40 else 75.0


def record_agent_telemetry(
    state: dict[str, Any],
    agent_name: str,
    started_at: float,
    token_breakdown: dict[str, int],
    accuracy_score: float,
) -> dict[str, Any]:
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    in_tok = int(token_breakdown.get("input_tokens", 0) or 0)
    out_tok = int(token_breakdown.get("output_tokens", 0) or 0)
    tot_tok = int(token_breakdown.get("total_tokens", in_tok + out_tok) or 0)
    cost_usd = compute_cost_usd(in_tok, out_tok)

    metrics_by_agent: dict[str, dict[str, Any]] = merge_agent_metrics(
        state.get("agent_metrics"),
        {
            agent_name: {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "tokens": tot_tok,
                "latency_ms": elapsed_ms,
                "latency_s": round(elapsed_ms / 1000, 2),
                "cost_usd": cost_usd,
                "cost_inr": round(cost_usd * USD_TO_INR, 4),
                "accuracy": round(float(accuracy_score), 1),
            }
        },
    )

    total_latency_ms = round(
        sum(float(m.get("latency_ms", 0.0)) for m in metrics_by_agent.values()), 1
    )
    total_cost_usd = round(
        sum(float(m.get("cost_usd", 0.0)) for m in metrics_by_agent.values()), 6
    )
    scores = [float(m.get("accuracy", 0.0)) for m in metrics_by_agent.values()]
    overall_accuracy = round(sum(scores) / len(scores), 1) if scores else 100.0

    emit_telemetry_event(
        "agent_metrics",
        f"{agent_name} metrics recorded.",
        {
            "agent": agent_name,
            "metrics": metrics_by_agent[agent_name],
            "overall": {
                "latency_ms": total_latency_ms,
                "latency_s": round(total_latency_ms / 1000, 2),
                "cost_usd": total_cost_usd,
                "cost_inr": round(total_cost_usd * USD_TO_INR, 4),
                "accuracy": overall_accuracy,
            },
        },
    )

    return {
        "agent_metrics": metrics_by_agent,
        "overall_latency_ms": total_latency_ms,
        "total_cost_usd": total_cost_usd,
        "total_cost_inr": round(total_cost_usd * USD_TO_INR, 4),
        "overall_accuracy": overall_accuracy,
    }