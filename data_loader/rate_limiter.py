import time
import threading
from collections import deque


class TokenBucketRateLimiter:
    """Strict token-bucket rate limiter enforcing TheStatsAPI QPS limits."""

    def __init__(self, max_qps: float = 2.0):
        self.max_qps = max_qps
        self.tokens = max_qps
        self.updated_at = time.monotonic()
        self.lock = threading.Lock()
        self.history = deque(maxlen=100)

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.tokens = min(self.max_qps, self.tokens + elapsed * self.max_qps)
            self.updated_at = now
            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.max_qps
                time.sleep(sleep_time)
                self.tokens = 0.0
                self.updated_at = time.monotonic()
            else:
                self.tokens -= 1.0
            self.history.append(time.monotonic())
