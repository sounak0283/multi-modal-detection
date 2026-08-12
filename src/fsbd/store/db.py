"""PostgreSQL event store.

Driver choice is a licence decision, not a preference
----------------------------------------------------
`psycopg2` is LGPL-with-exceptions and `psycopg` (v3) is LGPL-3.0. NOTICE.md excludes
LGPL and the CI gate fails on it, so the obvious driver would have broken this project's
own policy on the first `pip install`. `pg8000` is BSD-3-Clause, pure Python, DB-API 2.0,
and needs no compiler - which also removes a build dependency from the installer.

Connection handling
-------------------
pg8000 connections are not safe to share across threads. The alert thread writes while
the API thread reads, so connections are pooled and checked out one at a time. The pool
is small on purpose: this is one camera on one box, not a web service.

The store degrades rather than crashes. If PostgreSQL is unreachable the pipeline keeps
detecting and the dashboard keeps serving - it simply reports that persistence is down.
A database outage must not take the cameras offline.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pg8000.dbapi

log = logging.getLogger("fsbd.store")

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "sql" / "schema.sql"
SCHEMA_VERSION = 1
POOL_SIZE = 4
CONNECT_TIMEOUT_S = 10


@dataclass(frozen=True)
class Dsn:
    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str = "public"

    @property
    def safe(self) -> str:
        """Never log the password - this string ends up in support bundles."""
        suffix = f" (schema {self.schema})" if self.schema != "public" else ""
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}{suffix}"


IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def parse_dsn(url: str) -> Dsn:
    """postgresql://user:password@host:port/database[?schema=name]

    percent-decoded, because database passwords routinely contain characters that must
    be escaped in a URL and a silently wrong password is a confusing failure.

    The optional `schema` parameter lets this product share a database with something
    else without its tables mingling - a normal ask in corporate estates where a DBA
    hands out one database per team rather than one per application.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"postgresql", "postgres"}:
        raise ValueError(f"unsupported database scheme {parts.scheme!r}, expected postgresql://")
    if not parts.hostname:
        raise ValueError("database URL has no host")

    database = parts.path.lstrip("/")
    if not database:
        raise ValueError("database URL has no database name")

    schema = (parse_qs(parts.query).get("schema") or ["public"])[0]
    # Validated rather than quoted because it is interpolated into SET search_path,
    # which cannot take a bind parameter. An unvalidated value here would be SQL
    # injection through a config file.
    if not IDENTIFIER.match(schema):
        raise ValueError(
            f"invalid schema name {schema!r}: lowercase letters, digits and underscores only"
        )

    return Dsn(
        host=parts.hostname,
        port=parts.port or 5432,
        database=unquote(database),
        user=unquote(parts.username or ""),
        password=unquote(parts.password or ""),
        schema=schema,
    )


class Database:
    """Thread-safe PostgreSQL access with a fixed-size connection pool."""

    def __init__(self, url: str, pool_size: int = POOL_SIZE) -> None:
        self.dsn = parse_dsn(url)
        self._pool: queue.LifoQueue = queue.LifoQueue(maxsize=pool_size)
        self._all: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False

        for _ in range(pool_size):
            self._pool.put(self._connect())
        log.info("connected to %s", self.dsn.safe)

    def _connect(self):
        connection = pg8000.dbapi.connect(
            user=self.dsn.user,
            password=self.dsn.password or None,
            host=self.dsn.host,
            port=self.dsn.port,
            database=self.dsn.database,
            timeout=CONNECT_TIMEOUT_S,
        )
        connection.autocommit = False

        # Every connection, not just the first: pooled connections are handed to
        # whichever thread asks next, and a missing search_path would silently write to
        # public instead of the configured schema.
        if self.dsn.schema != "public":
            cursor = connection.cursor()
            cursor.execute(f"SET search_path TO {self.dsn.schema}, public")
            connection.commit()

        with self._lock:
            self._all.append(connection)
        return connection

    @contextmanager
    def connection(self):
        """Check a connection out of the pool, returning it even on error.

        A connection that raised is replaced rather than returned: pg8000 leaves a
        connection unusable after certain errors, and silently recycling a dead one
        turns a transient failure into a permanent one.
        """
        if self._closed:
            raise RuntimeError("database is closed")
        conn = self._pool.get()
        try:
            yield conn
        except Exception:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - already failing; nothing useful to do
                pass
            try:
                conn = self._connect()
            except Exception:
                log.exception("could not re-establish a database connection")
                raise
            raise
        finally:
            self._pool.put(conn)

    def close(self) -> None:
        self._closed = True
        with self._lock:
            for conn in self._all:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._all.clear()

    # -- schema ------------------------------------------------------------

    def migrate(self) -> None:
        """Apply sql/schema.sql. Idempotent - every statement is IF NOT EXISTS."""
        if not SCHEMA_PATH.is_file():
            raise FileNotFoundError(f"schema not found at {SCHEMA_PATH}")

        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connection() as conn:
            cursor = conn.cursor()
            if self.dsn.schema != "public":
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.dsn.schema}")
                cursor.execute(f"SET search_path TO {self.dsn.schema}, public")
            cursor.execute(sql)
            cursor.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                (SCHEMA_VERSION,),
            )
            conn.commit()
        log.info("schema applied (version %d)", SCHEMA_VERSION)

    def healthy(self) -> bool:
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                conn.rollback()
            return True
        except Exception:  # noqa: BLE001 - health must never raise
            return False

    # -- writes ------------------------------------------------------------

    def insert_event(self, event: dict[str, Any]) -> int:
        """Insert one alert and return its id.

        `ts` is stored as TIMESTAMPTZ in UTC. Passing a naive datetime would let the
        server guess a zone, and an alert log whose timestamps shift meaning is useless
        as evidence.
        """
        ts = event.get("ts")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=UTC)
        elif isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        elif ts is None:
            ts = datetime.now(UTC)

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (
                    ts, camera_id, kind, subtype, zone_id, zone_name, track_id,
                    message, severity, confidence, bbox, foot_point,
                    identity_status, identity_name, identity_confidence,
                    snapshot_path, clip_path
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                ) RETURNING id
                """,
                (
                    ts,
                    event["camera_id"],
                    event["kind"],
                    event.get("subtype"),
                    event.get("zone_id"),
                    event.get("zone_name"),
                    event.get("track_id"),
                    event["message"],
                    event.get("severity", "medium"),
                    event.get("confidence"),
                    _json(event.get("bbox")),
                    _json(event.get("foot_point")),
                    event.get("identity_status"),
                    event.get("identity_name"),
                    event.get("identity_confidence"),
                    event.get("snapshot_path"),
                    event.get("clip_path"),
                ),
            )
            event_id = cursor.fetchone()[0]
            conn.commit()
        return int(event_id)

    def mark_delivered(self, event_id: int, error: str | None = None) -> None:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE events
                   SET delivered = %s,
                       delivery_attempts = delivery_attempts + 1,
                       last_attempt_ts = now(),
                       delivery_error = %s
                 WHERE id = %s
                """,
                (error is None, error, event_id),
            )
            conn.commit()

    # -- reads -------------------------------------------------------------

    def recent_events(
        self,
        limit: int = 50,
        kind: str | None = None,
        zone_id: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if kind:
            clauses.append("kind = %s")
            params.append(kind)
        if zone_id:
            clauses.append("zone_id = %s")
            params.append(zone_id)
        if since:
            clauses.append("ts >= %s")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, ts, camera_id, kind, subtype, zone_id, zone_name, track_id,
                       message, severity, confidence, bbox, foot_point,
                       identity_status, identity_name, snapshot_path, clip_path,
                       delivered
                  FROM events {where}
                 ORDER BY ts DESC
                 LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
            columns = [c[0] for c in cursor.description]
            conn.rollback()  # read-only; do not hold a transaction open

        return [_row_to_dict(columns, row) for row in rows]

    def event_counts(self, since: datetime | None = None) -> dict[str, int]:
        with self.connection() as conn:
            cursor = conn.cursor()
            if since:
                cursor.execute(
                    "SELECT kind, count(*) FROM events WHERE ts >= %s GROUP BY kind", (since,)
                )
            else:
                cursor.execute("SELECT kind, count(*) FROM events GROUP BY kind")
            rows = cursor.fetchall()
            conn.rollback()
        return {str(kind): int(count) for kind, count in rows}

    def purge_older_than(self, cutoff: datetime) -> int:
        """Retention. PLAN.md section 10.3 requires a stated, enforced retention period."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE ts < %s", (cutoff,))
            removed = cursor.rowcount
            conn.commit()
        if removed:
            log.info("purged %d event(s) older than %s", removed, cutoff.isoformat())
        return int(removed)


def _json(value: Any) -> str | None:
    """Serialise a coordinate list, tolerating numpy scalars.

    Belt and braces with the float() casts upstream. Anything numeric that reaches here
    is a coordinate, and silently refusing to store an alert because a value is
    np.float32 rather than float is a terrible trade - the `default` hook converts
    rather than raising.
    """
    import json

    if value is None:
        return None
    return json.dumps(
        list(value) if isinstance(value, tuple) else value,
        default=lambda o: float(o) if hasattr(o, "__float__") else str(o),
    )


def _row_to_dict(columns: list[str], row: tuple) -> dict[str, Any]:
    import json

    record = dict(zip(columns, row, strict=True))
    for key in ("bbox", "foot_point"):
        value = record.get(key)
        if isinstance(value, str):
            try:
                record[key] = json.loads(value)
            except json.JSONDecodeError:
                record[key] = None
    if isinstance(record.get("ts"), datetime):
        ts = record["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        record["ts"] = ts.isoformat()
    return record
