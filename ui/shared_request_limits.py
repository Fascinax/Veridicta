"""Coordinate short rate transactions and kernel-held slots on one local host."""

from contextlib import closing, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from filelock import FileLock, SoftFileLock, Timeout

from ui.request_limits import MAX_OWNER_BUCKETS, WINDOW_SECONDS, RequestLimitExceeded, RequestPolicy

APPLICATION_ID = 0x56524451
SCHEMA_VERSION = 1
BUSY_TIMEOUT_SECONDS = 5
UNAVAILABLE_MESSAGE = "Le contrôle des quotas est indisponible. Réessayez plus tard."
RATE_MESSAGE = "Trop de demandes récentes. Réessayez dans une minute."


class RequestLimitsUnavailable(RequestLimitExceeded):
    pass


class SharedRequestLimits:
    def __init__(self, database: Path, policy: RequestPolicy = RequestPolicy()):
        self.database = Path(database).resolve()
        self.policy = policy
        self._policy_json = json.dumps(asdict(policy) | {
            "window_seconds": WINDOW_SECONDS, "owner_capacity": MAX_OWNER_BUCKETS,
        }, sort_keys=True, separators=(",", ":"))
        self._slot_directory = self.database.with_name(self.database.name + ".slots")
        try:
            self.database.parent.mkdir(parents=True, exist_ok=True)
            self._slot_directory.mkdir(exist_ok=True)
            with self._transaction() as connection:
                self._initialize(connection)
        except OSError:
            raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE) from None

    @contextmanager
    def _transaction(self):
        try:
            with closing(sqlite3.connect(self.database, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)) as connection:
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                    raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE)
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
        except (sqlite3.Error, OSError):
            raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE) from None

    def _initialize(self, connection) -> None:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if not tables and application_id == version == 0:
            for statement in (
                "CREATE TABLE policy (singleton INTEGER PRIMARY KEY CHECK(singleton=1), configuration TEXT NOT NULL)",
                "CREATE TABLE owners (owner TEXT PRIMARY KEY, last_seen REAL NOT NULL)",
                "CREATE TABLE events (owner TEXT NOT NULL, accepted_at REAL NOT NULL)",
                "CREATE INDEX events_owner_time ON events(owner, accepted_at)",
                "CREATE INDEX events_time ON events(accepted_at)",
            ):
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.execute("INSERT INTO policy VALUES (1, ?)", (self._policy_json,))
        self._require_policy(connection)
        if (tables and tables != {"policy", "owners", "events"}):
            raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE)

    def _require_policy(self, connection) -> None:
        if (connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
                or connection.execute("SELECT configuration FROM policy WHERE singleton=1").fetchone() != (self._policy_json,)):
            raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE)

    def _record_request(self, owner: str) -> None:
        owner_hash = hashlib.sha256(owner.encode("utf-8")).hexdigest()
        with self._transaction() as connection:
            self._require_policy(connection)
            now = time.time()
            if not math.isfinite(now):
                raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE)
            cutoff = now - WINDOW_SECONDS
            connection.execute("DELETE FROM events WHERE accepted_at <= ?", (cutoff,))
            connection.execute("DELETE FROM owners WHERE last_seen <= ?", (cutoff,))
            exists = connection.execute("SELECT 1 FROM owners WHERE owner=?", (owner_hash,)).fetchone()
            if not exists and connection.execute("SELECT COUNT(*) FROM owners").fetchone()[0] >= MAX_OWNER_BUCKETS:
                raise RequestLimitExceeded(RATE_MESSAGE)
            recent = connection.execute("SELECT COUNT(*) FROM events WHERE owner=?", (owner_hash,)).fetchone()[0]
            if recent >= self.policy.requests_per_minute:
                raise RequestLimitExceeded(RATE_MESSAGE)
            connection.execute("INSERT INTO owners VALUES (?, ?) ON CONFLICT(owner) DO UPDATE SET last_seen=MAX(owners.last_seen, excluded.last_seen)", (owner_hash, now))
            connection.execute("INSERT INTO events VALUES (?, ?)", (owner_hash, now))

    def _reserve_slot(self):
        try:
            for position in range(self.policy.max_concurrent):
                slot = FileLock(self._slot_directory / f"slot-{position}.lock")
                if isinstance(slot, SoftFileLock):
                    raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE)
                try:
                    slot.acquire(timeout=0)
                except Timeout:
                    continue
                if isinstance(slot, SoftFileLock):
                    slot.release()
                    raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE)
                return slot
        except OSError:
            raise RequestLimitsUnavailable(UNAVAILABLE_MESSAGE) from None
        raise RequestLimitExceeded("Le service traite déjà plusieurs demandes. Réessayez dans un instant.")

    @contextmanager
    def admit(self, owner: str, query: str):
        if not isinstance(owner, str) or not owner:
            raise RequestLimitExceeded("Le périmètre de la demande est invalide.")
        if not isinstance(query, str) or len(query) > self.policy.max_query_chars:
            raise RequestLimitExceeded(f"La question est trop longue. Limite : {self.policy.max_query_chars} caractères.")
        slot = self._reserve_slot()
        try:
            self._record_request(owner)
            yield
        finally:
            slot.release()


def request_limits_from_environment() -> SharedRequestLimits:
    configured = os.getenv("VERIDICTA_REQUEST_LIMITS_DB")
    database = Path(configured) if configured else Path(os.getenv("VERIDICTA_WORKSPACE_DIR", "data/cases")) / ".request-limits.sqlite3"
    return SharedRequestLimits(database)
