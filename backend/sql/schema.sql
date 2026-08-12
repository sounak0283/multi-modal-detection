-- Fire, Smoke & Boundary Detection - PostgreSQL schema.
--
-- Applied automatically at startup by src/fsbd/store/db.py; kept here as the readable
-- canonical reference and for DBAs who want to review before deployment.
--
-- Driver note: this project uses pg8000 (BSD-3-Clause, pure Python), NOT psycopg2.
-- psycopg2 is LGPL-with-exceptions and psycopg3 is LGPL-3.0, both of which the licence
-- policy in NOTICE.md excludes and the CI gate would fail on. See NOTICE.md section 3.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- persons - the extension point for the face-recognition layer.
--
-- Deliberately created now, empty, rather than left as a future migration. The
-- identity columns on `events` reference it, so wiring YuNet + SFace later becomes
-- "populate this table and set three columns" instead of a schema change against a
-- table that by then holds live alert history.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS persons (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    external_id TEXT,                       -- employee number, badge id, etc.
    active      BOOLEAN     NOT NULL DEFAULT true,
    -- Embeddings are NOT stored here. They are biometric data under GDPR Art. 9,
    -- India's DPDP Act and Illinois BIPA, and need their own consent and retention
    -- story (PLAN.md section 10.3). Kept in a separate store so this table can be
    -- deployed with the identity feature switched off.
    enrolled_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persons_external_id
    ON persons (external_id) WHERE external_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- events - every alert the system has ever raised.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id            BIGSERIAL   PRIMARY KEY,
    -- TIMESTAMPTZ, not TIMESTAMP: an alert log that shifts meaning across a DST
    -- boundary is useless as evidence, which is part of what this product is for.
    ts            TIMESTAMPTZ NOT NULL,
    camera_id     TEXT        NOT NULL,
    kind          TEXT        NOT NULL,   -- boundary | fire | smoke | health
    subtype       TEXT,                   -- entry | exit | feed_lost | ...
    zone_id       TEXT,
    zone_name     TEXT,
    track_id      INTEGER,
    -- The operator-facing sentence, stored rather than rebuilt at read time so the
    -- history always shows what was actually sent, even after the wording changes.
    message       TEXT        NOT NULL,
    severity      TEXT        NOT NULL DEFAULT 'medium',
    confidence    REAL,
    bbox          JSONB,                  -- [x1, y1, x2, y2] normalised 0-1
    foot_point    JSONB,                  -- [x, y] normalised 0-1

    -- Identity layer - all nullable, all unused until the face layer ships.
    person_id           BIGINT REFERENCES persons (id) ON DELETE SET NULL,
    identity_status     TEXT,             -- known | unknown_face | no_face
    identity_name       TEXT,
    identity_confidence REAL,

    snapshot_path TEXT,
    clip_path     TEXT,

    -- Delivery tracking. A Telegram outage must not silently swallow a fire alert,
    -- and must not spin forever either.
    delivered         BOOLEAN     NOT NULL DEFAULT false,
    delivery_attempts INTEGER     NOT NULL DEFAULT 0,
    last_attempt_ts   TIMESTAMPTZ,
    delivery_error    TEXT,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_ts        ON events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events (camera_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind_ts   ON events (kind, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_zone_ts   ON events (zone_id, ts DESC);

-- Partial index: the retry loop only ever scans undelivered rows, and on a busy site
-- that is a handful out of millions.
CREATE INDEX IF NOT EXISTS idx_events_undelivered
    ON events (ts) WHERE delivered = false;
