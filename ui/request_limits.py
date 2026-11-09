"""Bound interactive requests per process and per workspace owner."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
import time

WINDOW_SECONDS = 60
MAX_OWNER_BUCKETS = 10000


class RequestLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class RequestPolicy:
    max_concurrent: int = 2
    requests_per_minute: int = 10
    max_query_chars: int = 6000

    def __post_init__(self):
        values = (self.max_concurrent, self.requests_per_minute, self.max_query_chars)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("request limits must be positive")


class RequestLimits:
    def __init__(self, policy: RequestPolicy = RequestPolicy()):
        self.policy = policy
        self._slots = BoundedSemaphore(policy.max_concurrent)
        self._lock = Lock()
        self._requests: dict[str, list[float]] = {}

    @contextmanager
    def admit(self, owner: str, query: str):
        if len(query) > self.policy.max_query_chars:
            raise RequestLimitExceeded(f"La question est trop longue. Limite : {self.policy.max_query_chars} caractères.")
        if not self._slots.acquire(blocking=False):
            raise RequestLimitExceeded("Le service traite déjà plusieurs demandes. Réessayez dans un instant.")
        try:
            with self._lock:
                now = time.monotonic()
                self._requests = {key: [stamp for stamp in stamps if stamp > now - WINDOW_SECONDS]
                                  for key, stamps in self._requests.items() if stamps and stamps[-1] > now - WINDOW_SECONDS}
                recent = self._requests.get(owner, [])
                if len(recent) >= self.policy.requests_per_minute or (
                    owner not in self._requests and len(self._requests) >= MAX_OWNER_BUCKETS
                ):
                    raise RequestLimitExceeded("Trop de demandes récentes. Réessayez dans une minute.")
                self._requests[owner] = [*recent, now]
            yield
        finally:
            self._slots.release()
