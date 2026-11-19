
import time
import uuid
import threading

from collections import deque
from threading import BoundedSemaphore


class HybridTokenRateLimiter:

    def __init__(
        self,
        tpm_limit=8000,
        window_seconds=60,
        max_concurrent=2,
    ):

        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds

        self.bucket_capacity = tpm_limit
        self.refill_rate = tpm_limit / window_seconds
        self.bucket_tokens = float(tpm_limit)

        self.last_refill = time.monotonic()

        self.events = deque()
        self.lock = threading.Lock()

        self.semaphore = BoundedSemaphore(
            max_concurrent
        )

    def _refill(self, now):

        elapsed = max(
            0,
            now - self.last_refill
        )

        self.bucket_tokens = min(
            self.bucket_capacity,
            self.bucket_tokens + elapsed * self.refill_rate
        )

        self.last_refill = now

    def _cleanup(self, now):

        cutoff = now - self.window_seconds

        while self.events and self.events[0][0] <= cutoff:
            self.events.popleft()

    def acquire(self, tokens):

        tokens = max(1, int(tokens))

        if tokens > self.tpm_limit:
            raise ValueError("Request exceeds TPM capacity.")

        while True:

            now = time.monotonic()

            with self.lock:

                self._refill(now)
                self._cleanup(now)

                usage = sum(
                    event[2] for event in self.events
                )

                window_wait = 0
                bucket_wait = 0

                if usage + tokens > self.tpm_limit:

                    required = usage + tokens - self.tpm_limit
                    released = 0

                    for timestamp, _, amount in self.events:

                        released += amount

                        if released >= required:
                            window_wait = max(
                                0,
                                timestamp + self.window_seconds - now
                            )
                            break

                if self.bucket_tokens < tokens:
                    bucket_wait = (
                        tokens - self.bucket_tokens
                    ) / self.refill_rate

                wait_time = max(window_wait, bucket_wait)

                if wait_time <= 0:

                    reservation_id = uuid.uuid4().hex

                    self.bucket_tokens -= tokens

                    self.events.append(
                        (now, reservation_id, tokens)
                    )

                    return reservation_id

            time.sleep(max(0.05, wait_time))

    def settle(self, reservation_id, actual_tokens):

        now = time.monotonic()
        actual_tokens = max(0, int(actual_tokens))

        with self.lock:

            self._refill(now)
            self._cleanup(now)

            for index, event in enumerate(self.events):

                timestamp, event_id, reserved = event

                if event_id == reservation_id:

                    difference = reserved - actual_tokens

                    self.bucket_tokens = max(
                        0,
                        min(
                            self.bucket_capacity,
                            self.bucket_tokens + difference
                        )
                    )

                    self.events[index] = (
                        timestamp,
                        event_id,
                        actual_tokens
                    )

                    return True

            return False

    def release(self, reservation_id):

        now = time.monotonic()

        with self.lock:

            self._refill(now)

            for index, event in enumerate(self.events):

                timestamp, event_id, reserved = event

                if event_id == reservation_id:

                    self.bucket_tokens = min(
                        self.bucket_capacity,
                        self.bucket_tokens + reserved
                    )

                    del self.events[index]
                    break

            self._cleanup(now)

    def usage(self):

        now = time.monotonic()

        with self.lock:

            self._refill(now)
            self._cleanup(now)

            return sum(
                event[2] for event in self.events
            )
