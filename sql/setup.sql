-- One-time database and role setup. Run as a PostgreSQL superuser:
--
--   psql -U postgres -f sql/setup.sql
--
-- Then put the URL in .env (git-ignored):
--
--   FSBD_DATABASE_URL=postgresql://fsbd_app:CHANGE_THIS@localhost:5432/fsbd
--
-- The application role deliberately has NO createdb and NO superuser. It only ever
-- needs to read and write its own tables, and a compromised dashboard - which has no
-- authentication - should not be able to reach anything else on the server.

-- Change this before running. A default password in a commercial deployment is a
-- finding waiting to happen.
CREATE ROLE fsbd_app WITH LOGIN PASSWORD 'CHANGE_THIS';

CREATE DATABASE fsbd OWNER fsbd_app;

\connect fsbd

-- The application applies sql/schema.sql itself at startup, so nothing further is
-- required here. Granting on the public schema keeps that working on PostgreSQL 15+,
-- where PUBLIC lost its default CREATE privilege.
GRANT ALL ON SCHEMA public TO fsbd_app;
