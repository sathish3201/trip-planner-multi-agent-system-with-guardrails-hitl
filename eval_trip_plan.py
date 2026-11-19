"""
eval.py — Evaluation harness for the LangGraph multi-agent travel planner.

Usage:
    python eval.py --dataset evation.json
    python eval.py --dataset evation.json --module app.backend --workers 2
    python eval.py --dataset-dir eval_datasets --output-dir evaluation_results
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENT_NODES = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

EXPECTED_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate evaluation cases from a JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("test_cases") or data.get("cases") or data.get("records")
    else:
        raise ValueError("Dataset must be a JSON array or object.")

    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No test cases found in '{path}'.")

    required_fields = {
        "name",
        "query",
        "required_agents",
        "forbidden_agents",
        "required_terms",
        "forbidden_terms",
        "should_block",
        "should_require_approval",
        "auto_approve",
    }

    seen_names: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Test case {index} in '{path.name}' must be an object.")

        missing = required_fields - set(case.keys())
        if missing:
            raise ValueError(
                f"Test case {index} in '{path.name}' is missing fields: {sorted(missing)}"
            )

        name = case["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Test case {index} in '{path.name}' has an invalid name.")
        if name in seen_names:
            raise ValueError(f"Duplicate test case name '{name}' in '{path.name}'.")
        seen_names.add(name)

        if not isinstance(case["query"], str) or not case["query"].strip():
            raise ValueError(f"Test case '{name}' has an empty query.")

        for list_field in (
            "required_agents",
            "forbidden_agents",
            "required_terms",
            "forbidden_terms",
        ):
            if not isinstance(case[list_field], list):
                raise ValueError(f"'{list_field}' must be a list in '{name}'.")

        for bool_field in (
            "should_block",
            "should_require_approval",
            "auto_approve",
        ):
            if not isinstance(case[bool_field], bool):
                raise ValueError(f"'{bool_field}' must be boolean in '{name}'.")

    return cases


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def contains_all(text: str, terms: list[str]) -> bool:
    lowered = text.casefold()
    return all(term.casefold() in lowered for term in terms)


def ordered_agent_nodes(trajectory: list[str]) -> list[str]:
    return [node for node in trajectory if node in AGENT_NODES]


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    pos = 0
    for item in actual:
        if pos < len(expected) and item == expected[pos]:
            pos += 1
    return pos == len(expected)


def load_backend(module_name: str):
    try:
        backend = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import backend module '{module_name}'. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(backend, "travel_graph"):
        raise RuntimeError(
            f"Backend module '{module_name}' is missing compiled 'travel_graph'."
        )

    return backend


def make_initial_state(query: str, backend) -> dict[str, Any]:
    if hasattr(backend, "create_initial_state"):
        return backend.create_initial_state(query)

    from langchain_core.messages import HumanMessage

    token_limit = int(getattr(backend, "TOKEN_LIMIT", 7500))
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": {},
        "supervisor_reasoning": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": 0,
        "token_usage": 0,
        "token_limit": token_limit,
    }


def extract_update_nodes(update: Any) -> list[str]:
    if not isinstance(update, dict):
        return []
    return [str(key) for key in update.keys() if not str(key).startswith("__")]


def snapshot_has_interrupt(snapshot: Any) -> bool:
    if getattr(snapshot, "interrupts", None):
        return True
    for task in getattr(snapshot, "tasks", ()) or ():
        if getattr(task, "interrupts", None):
            return True
    return False


def run_graph_case(backend, case: dict[str, Any]) -> dict[str, Any]:
    graph = backend.travel_graph
    thread_id = f"eval_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = make_initial_state(case["query"], backend)

    trajectory: list[str] = []
    started = time.perf_counter()
    interrupted = False

    try:
        for update in graph.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            trajectory.extend(extract_update_nodes(update))

        snapshot = graph.get_state(config)
        interrupted = snapshot_has_interrupt(snapshot)

        if interrupted:
            resume_value = {
                "approved": bool(case.get("auto_approve", True)),
                "feedback": "",
            }
            for update in graph.stream(
                backend.Command(resume=resume_value),
                config=config,
                stream_mode="updates",
            ):
                trajectory.extend(extract_update_nodes(update))

        final_snapshot = graph.get_state(config)
        final_state = dict(getattr(final_snapshot, "values", {}) or {})

    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "case": case["name"],
            "query": case["query"],
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": round(elapsed, 3),
            "trajectory": trajectory,
            "interrupted": interrupted,
            "final_state": {},
        }

    elapsed = time.perf_counter() - started
    if interrupted and "human_approval" not in trajectory:
        trajectory.append("human_approval")

    return {
        "case": case["name"],
        "query": case["query"],
        "error": None,
        "latency_seconds": round(elapsed, 3),
        "trajectory": trajectory,
        "interrupted": interrupted,
        "final_state": final_state,
    }


def evaluate_case(
    case: dict[str, Any],
    run: dict[str, Any],
    max_latency: float,
) -> dict[str, Any]:
    state = run.get("final_state", {})
    trajectory = run.get("trajectory", [])
    answer = str(state.get("final_response") or state.get("itinerary") or "")
    selected = list(state.get("selected_agents") or [])
    executed_agents = ordered_agent_nodes(trajectory)

    required_agents = case.get("required_agents", [])
    forbidden_agents = case.get("forbidden_agents", [])

    selection_pass = (
        all(agent in selected for agent in required_agents)
        and all(agent not in selected for agent in forbidden_agents)
    )

    trajectory_pass = (
        all(agent in executed_agents for agent in required_agents)
        and all(agent not in executed_agents for agent in forbidden_agents)
        and is_subsequence(
            [agent for agent in EXPECTED_ORDER if agent in executed_agents],
            executed_agents,
        )
    )

    if case["should_block"]:
        guardrail_pass = (
            state.get("guardrail_allowed") is False
            and "guardrail_blocked" in trajectory
            and not executed_agents
        )
    else:
        guardrail_pass = state.get("guardrail_allowed") is True

    required_terms = case.get("required_terms", [])
    forbidden_terms = case.get("forbidden_terms", [])
    required_hits = sum(
        1 for term in required_terms if term.casefold() in answer.casefold()
    )
    accuracy_score = (
        100.0
        if not required_terms
        else round(100.0 * required_hits / len(required_terms), 2)
    )

    budget_exhausted = "insufficient token budget" in answer.casefold()
    answer_quality_pass = (
        bool(answer.strip())
        and not budget_exhausted
        and contains_all(answer, required_terms)
        and not contains_any(answer, forbidden_terms)
    )

    approval_pass = (
        run.get("interrupted") is bool(case["should_require_approval"])
    )

    latency = float(run.get("latency_seconds", 0.0))
    latency_pass = latency <= max_latency

    llm_calls = int(state.get("llm_calls", 0) or 0)
    token_usage = int(state.get("token_usage", 0) or 0)
    token_limit = int(state.get("token_limit", 0) or 0)

    reliability_pass = (
        not run.get("error")
        and not budget_exhausted
        and bool(state)
        and (case["should_block"] or bool(answer.strip()))
        and (token_limit <= 0 or token_usage <= token_limit)
    )

    metrics = {
        "answer_quality": answer_quality_pass,
        "agent_selection": selection_pass,
        "agent_trajectory": trajectory_pass,
        "latency": latency_pass,
        "safety_guardrail": guardrail_pass,
        "human_approval": approval_pass,
        "reliability": reliability_pass,
    }

    passed = sum(bool(value) for value in metrics.values())
    score = round(100.0 * passed / len(metrics), 2)

    return {
        "case": case["name"],
        "score": score,
        "metrics": metrics,
        "budget_exhausted": budget_exhausted,
        "accuracy_score": accuracy_score,
        "required_terms_found": required_hits,
        "required_terms_total": len(required_terms),
        "latency_seconds": latency,
        "latency_pass": latency_pass,
        "llm_calls": llm_calls,
        "token_usage": token_usage,
        "token_limit": token_limit,
        "selected_agents": selected,
        "executed_agents": executed_agents,
        "trajectory": trajectory,
        "guardrail_allowed": state.get("guardrail_allowed"),
        "approval_interrupt_observed": run.get("interrupted"),
        "answer_preview": answer[:1200],
        "error": run.get("error"),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"total_cases": 0, "overall_score": 0}

    metric_names = list(results[0]["metrics"].keys())
    metric_pass_rates = {
        name: round(
            100.0 * sum(bool(row["metrics"][name]) for row in results) / len(results),
            2,
        )
        for name in metric_names
    }

    latencies = [row["latency_seconds"] for row in results]
    token_counts = [row["token_usage"] for row in results]
    llm_calls = [row["llm_calls"] for row in results]
    scores = [row["score"] for row in results]

    return {
        "total_cases": len(results),
        "overall_score": round(statistics.mean(scores), 2),
        "metric_pass_rates_percent": metric_pass_rates,
        "budget_exhausted_cases": sum(bool(row.get("budget_exhausted")) for row in results),
        "average_latency_seconds": round(statistics.mean(latencies), 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "max_latency_seconds": round(max(latencies), 3),
        "average_token_usage": round(statistics.mean(token_counts), 2),
        "average_llm_calls": round(statistics.mean(llm_calls), 2),
        "total_errors": sum(bool(row["error"]) for row in results),
    }


def write_reports(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    run_number: int,
    dataset_label: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in dataset_label)
    stem = f"eval_{safe_label}_run{run_number}_{stamp}"

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_label,
        "summary": summary,
        "results": results,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    flat_rows = []
    for row in results:
        flat = {
            "case": row["case"],
            "score": row["score"],
            "budget_exhausted": row["budget_exhausted"],
            "accuracy_score": row["accuracy_score"],
            "latency_seconds": row["latency_seconds"],
            "llm_calls": row["llm_calls"],
            "token_usage": row["token_usage"],
            "selected_agents": "|".join(row["selected_agents"]),
            "executed_agents": "|".join(row["executed_agents"]),
            "trajectory": "|".join(row["trajectory"]),
            "error": row["error"] or "",
        }
        flat.update({f"pass_{k}": v for k, v in row["metrics"].items()})
        flat_rows.append(flat)

    if flat_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    return json_path, csv_path


def print_summary(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("TRAVEL AGENT EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Test cases             : {summary['total_cases']}")
    print(f"Overall score          : {summary['overall_score']}%")
    print(f"Budget exhausted       : {summary['budget_exhausted_cases']}")
    print(f"Average latency        : {summary['average_latency_seconds']} sec")
    print(f"Median latency         : {summary['median_latency_seconds']} sec")
    print(f"Maximum latency        : {summary['max_latency_seconds']} sec")
    print(f"Average token usage    : {summary['average_token_usage']}")
    print(f"Average LLM calls      : {summary['average_llm_calls']}")
    print(f"Execution errors       : {summary['total_errors']}")

    print("\nMetric pass rates")
    for metric, rate in summary["metric_pass_rates_percent"].items():
        print(f"  {metric:22s}: {rate:6.2f}%")

    print("\nPer-case results")
    for row in results:
        status = "PASS" if row["score"] == 100 else "CHECK"
        print(
            f"  [{status:5s}] {row['case']:32s} "
            f"score={row['score']:6.2f}% "
            f"tokens={row['token_usage']:4d} "
            f"latency={row['latency_seconds']:6.2f}s"
        )
        if row["budget_exhausted"]:
            print("         WARNING: Insufficient token budget in answer")
        if row["error"]:
            print(f"         ERROR: {row['error']}")
        print(f"         trajectory: {' -> '.join(row['trajectory']) or '(none)'}")

    print("=" * 72)


def run_and_evaluate_case(backend, case: dict[str, Any], max_latency: float) -> dict[str, Any]:
    raw_run = run_graph_case(backend, case)
    return evaluate_case(case, raw_run, max_latency=max_latency)


def run_suite(
    backend,
    cases: list[dict[str, Any]],
    workers: int,
    max_latency: float,
    output_dir: Path,
    run_number: int,
    dataset_label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"progress_{dataset_label}_run{run_number}.jsonl"
    completed_count = 0
    progress_lock = threading.Lock()
    indexed_results: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="travel-eval",
    ) as executor:
        future_to_case = {
            executor.submit(run_and_evaluate_case, backend, case, max_latency): (idx, case)
            for idx, case in enumerate(cases)
        }

        for future in as_completed(future_to_case):
            idx, case = future_to_case[future]
            try:
                evaluated = future.result()
            except Exception as exc:
                evaluated = {
                    "case": case["name"],
                    "score": 0.0,
                    "metrics": {},
                    "budget_exhausted": False,
                    "accuracy_score": 0.0,
                    "required_terms_found": 0,
                    "required_terms_total": len(case.get("required_terms", [])),
                    "latency_seconds": 0.0,
                    "latency_pass": False,
                    "llm_calls": 0,
                    "token_usage": 0,
                    "token_limit": 0,
                    "selected_agents": [],
                    "executed_agents": [],
                    "trajectory": [],
                    "guardrail_allowed": None,
                    "approval_interrupt_observed": False,
                    "answer_preview": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }

            indexed_results[idx] = evaluated
            completed_count += 1

            with progress_lock:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(evaluated, ensure_ascii=False, default=str) + "\n")
                    handle.flush()

                print(
                    f"[{completed_count}/{len(cases)}] "
                    f"{case['name']} | "
                    f"Score={evaluated['score']}% | "
                    f"Tokens={evaluated['token_usage']} | "
                    f"Latency={evaluated['latency_seconds']}s | "
                    f"Error={evaluated['error']}",
                    flush=True,
                )

    results = [indexed_results[idx] for idx in sorted(indexed_results)]
    summary = summarize(results)
    print_summary(summary, results)
    json_path, csv_path = write_reports(
        results,
        summary,
        output_dir,
        run_number,
        dataset_label,
    )
    print(f"JSON report : {json_path}")
    print(f"CSV report  : {csv_path}")
    return {"dataset": dataset_label, "summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the LangGraph multi-agent travel planner."
    )
    parser.add_argument(
        "--dataset",
        default="evation.json",
        help="Path to a JSON evaluation file (default: evation.json).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional directory of multiple JSON evaluation files to run sequentially.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent test cases (default: 2).",
    )
    parser.add_argument(
        "--module",
        default=os.getenv("TRAVEL_AGENT_MODULE", "app.backend"),
        help="Backend module containing travel_graph (default: app.backend).",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=float(os.getenv("EVAL_MAX_LATENCY_SECONDS", "180")),
        help="Maximum acceptable latency per case in seconds.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation repetitions.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory for JSON, CSV, and JSONL progress reports.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only a named test case; repeatable.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_latency <= 0:
        parser.error("--max-latency must be greater than zero")

    try:
        backend = load_backend(args.module)
    except Exception as exc:
        print(f"[SETUP ERROR] {exc}", file=sys.stderr)
        return 2

    if args.dataset_dir:
        dataset_files = sorted(Path(args.dataset_dir).glob("*.json"))
        if not dataset_files:
            print(f"[DATASET ERROR] No .json files in '{args.dataset_dir}'", file=sys.stderr)
            return 2
    else:
        dataset_files = [Path(args.dataset)]

    output_dir = Path(args.output_dir)

    for ds_path in dataset_files:
        try:
            cases = load_dataset(ds_path)
        except Exception as exc:
            print(f"[DATASET ERROR] {exc}", file=sys.stderr)
            return 2

        if args.cases:
            available = {c["name"]: c for c in cases}
            unknown = set(args.cases) - set(available)
            if unknown:
                parser.error(f"Unknown test case(s): {', '.join(sorted(unknown))}")
            cases = [available[name] for name in args.cases]

        print(f"\nLoaded {len(cases)} cases from {ds_path}.")
        for run_number in range(1, args.runs + 1):
            print(f"\nStarting run {run_number}/{args.runs} for {ds_path.name}")
            run_suite(
                backend=backend,
                cases=cases,
                workers=args.workers,
                max_latency=args.max_latency,
                output_dir=output_dir,
                run_number=run_number,
                dataset_label=ds_path.stem,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())