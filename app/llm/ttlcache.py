import copy
import hashlib
import json
import threading
import time
from typing import Any


class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL and max-capacity eviction."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 512):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _evict_expired_locked(self, now: float) -> None:
        expired_keys = [
            key for key, (expires_at, _) in self._store.items() if expires_at <= now
        ]
        for key in expired_keys:
            del self._store[key]

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= now:
                del self._store[key]
                return None

            return copy.deepcopy(value)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            return

        now = time.monotonic()
        copied = copy.deepcopy(value)

        with self._lock:
            self._evict_expired_locked(now)

            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self.max_entries:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]

            self._store[key] = (now + ttl, copied)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def make_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        {"namespace": namespace, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def is_valid_llm_response(response: Any) -> bool:
    """Ensure the stream assembled into a non-empty response with a valid choice."""
    if response is None:
        return False

    choices = getattr(response, "choices", None)
    if not choices or not isinstance(choices, (list, tuple)):
        return False

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        return False

    # Optional: if finish_reason is present and indicates length truncation or error,
    # decide whether you want to cache it. Usually "stop" or "tool_calls" is safe.
    finish_reason = getattr(first_choice, "finish_reason", None)
    if finish_reason in ("error", "content_filter"):
        return False

    return True


def is_valid_agent_text(text: Any, error_prefixes: tuple[str, ...] = ()) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return not any(stripped.startswith(prefix) for prefix in error_prefixes)