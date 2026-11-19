from typing import Any


def list_checkpoints(graph, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent checkpoints for a thread (newest first)."""
    config = {"configurable": {"thread_id": thread_id}}
    history = []

    for snapshot in graph.get_state_history(config, limit=limit):
        history.append({
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "next": snapshot.next,
            "created_at": snapshot.created_at,
            "parent_checkpoint_id": (
                snapshot.parent_config["configurable"].get("checkpoint_id")
                if snapshot.parent_config
                else None
            ),
            "approved": snapshot.values.get("approved"),
            "has_itinerary": bool(snapshot.values.get("itinerary")),
            "has_final_response": bool(snapshot.values.get("final_response")),
        })

    return history


def get_checkpoint_state(graph, thread_id: str, checkpoint_id: str):
    """Fetch a specific state snapshot by checkpoint_id."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    return graph.get_state(config)


def fork_and_resume(
    graph,
    thread_id: str,
    checkpoint_id: str,
    state_updates: dict[str, Any],
    as_node: str | None = None,
    stream_mode: str = "values",
):
    """
    Fork from `checkpoint_id` with `state_updates` and run forward from that point.

    - Pass `as_node="human_approval"` when overriding state as if `human_approval`
      just produced those values (e.g., {"approved": False, "human_feedback": "..."}).
    - Pass `as_node=None` (default) when modifying state before a node runs.
    """
    base_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    fork_config = graph.update_state(
        base_config,
        values=state_updates,
        as_node=as_node,
    )

    # Passing `None` as input resumes/continues from the forked checkpoint
    # using the graph's existing saved state.
    return graph.invoke(None, config=fork_config, stream_mode=stream_mode)