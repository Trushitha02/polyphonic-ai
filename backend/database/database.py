import os
import re
import sqlite3
import sys
import threading
from urllib.parse import urlparse

try:
    import mysql.connector
except ImportError:
    mysql = None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None


# ---------------------------------------------------------------------------
# PostgreSQL Compatibility Wrappers
# ---------------------------------------------------------------------------

class PostgresCursorWrapper:
    """Wraps psycopg2 cursor to provide MySQL-compatible lastrowid and dict results."""

    def __init__(self, raw_cursor):
        self._cursor = raw_cursor
        self.lastrowid = None

    def execute(self, query, params=None):
        clean_query = query.strip()
        is_insert = clean_query.upper().startswith("INSERT INTO")

        # Automatically append RETURNING for tables where routes need cursor.lastrowid
        if is_insert and "RETURNING" not in clean_query.upper():
            lowered = clean_query.lower()
            pk_map = {
                "users": "user_id",
                "audio_files": "audio_id",
                "performances": "performance_id",
                "progress": "progress_id",
                "instrument_detections": "detection_id",
                "separation_results": "separation_id",
                "practice_recommendations": "recommendation_id",
                "transcriptions": "transcription_id",
            }
            for tbl, col in pk_map.items():
                if f"into {tbl}" in lowered or f"into `{tbl}`" in lowered or f'into "{tbl}"' in lowered:
                    query = f"{query.rstrip(';')} RETURNING {col}"
                    break

        if params is not None:
            self._cursor.execute(query, params)
        else:
            self._cursor.execute(query)

        if is_insert and "RETURNING" in query.upper():
            try:
                row = self._cursor.fetchone()
                if row:
                    if isinstance(row, dict):
                        self.lastrowid = next(iter(row.values()), None)
                    elif isinstance(row, (tuple, list)):
                        self.lastrowid = row[0]
            except Exception:
                pass

        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        if row and isinstance(row, dict):
            return dict(row)
        return row

    def fetchall(self):
        rows = self._cursor.fetchall()
        if rows and isinstance(rows[0], dict):
            return [dict(r) for r in rows]
        return rows

    def fetchmany(self, size=None):
        return self._cursor.fetchmany(size) if size else self._cursor.fetchmany()

    def close(self):
        return self._cursor.close()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class PostgresConnectionWrapper:
    """Wraps psycopg2 connection to mimic MySQL connector interface."""

    def __init__(self, raw_conn):
        self._conn = raw_conn

    def cursor(self, dictionary=False):
        if dictionary and psycopg2:
            raw_cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            raw_cur = self._conn.cursor()
        return PostgresCursorWrapper(raw_cur)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def is_connected(self):
        return getattr(self._conn, "closed", 1) == 0

    def __getattr__(self, name):
        return getattr(self._conn, name)


# ---------------------------------------------------------------------------
# SQLite Compatibility Wrappers (zero-config local fallback)
# ---------------------------------------------------------------------------
# Used automatically when no MySQL/PostgreSQL server is configured or
# reachable, so the whole app (login, uploads, separation, history)
# works out of the box on any machine.  Set DB_TYPE=sqlite to force it.

SQLITE_PATH = os.environ.get(
    "SQLITE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "polyphonic.db"),
)

_sqlite_fallback_active = False
_fallback_reason = None
_fallback_lock = threading.Lock()


class SQLiteCursorWrapper:
    """Accepts MySQL-style %s placeholders and optional dict rows."""

    def __init__(self, raw_cursor, dictionary=False):
        self._cursor = raw_cursor
        self._dictionary = dictionary

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def execute(self, query, params=None):
        query = query.replace("%s", "?")
        if query.strip().upper().startswith("SHOW TABLES"):
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        if params is None:
            self._cursor.execute(query)
        else:
            self._cursor.execute(query, tuple(params))
        return self

    def _convert(self, row):
        if row is None:
            return None
        if self._dictionary:
            return {key: row[key] for key in row.keys()}
        return tuple(row)

    def fetchone(self):
        return self._convert(self._cursor.fetchone())

    def fetchall(self):
        return [self._convert(r) for r in self._cursor.fetchall()]

    def fetchmany(self, size=None):
        rows = self._cursor.fetchmany(size) if size else self._cursor.fetchmany()
        return [self._convert(r) for r in rows]

    def close(self):
        return self._cursor.close()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class SQLiteConnectionWrapper:
    def __init__(self, raw_conn):
        self._conn = raw_conn

    def cursor(self, dictionary=False):
        return SQLiteCursorWrapper(self._conn.cursor(), dictionary=dictionary)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def is_connected(self):
        return True

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _sqlite_connection():
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    wrapped = SQLiteConnectionWrapper(conn)
    _ensure_sqlite_schema(wrapped)
    return wrapped


_sqlite_schema_ready = False


def _ensure_sqlite_schema(conn):
    global _sqlite_schema_ready
    if _sqlite_schema_ready:
        return
    schema_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_sqlite.sql")
    with open(schema_file, "r", encoding="utf-8") as f:
        conn._conn.executescript(f.read())
    conn.commit()
    _sqlite_schema_ready = True


def _db_explicitly_configured():
    return bool(
        os.environ.get("DB_TYPE", "").strip()
        or os.environ.get("DATABASE_URL", "").strip()
        or os.environ.get("DB_HOST", "").strip()
    )


# ---------------------------------------------------------------------------
# Database Type Detection
# ---------------------------------------------------------------------------

def get_fallback_reason():
    return _fallback_reason


def get_db_type():
    """Detect whether database is PostgreSQL or MySQL."""
    explicit_type = os.environ.get("DB_TYPE", "").strip().lower()
    if explicit_type in ("sqlite", "sqlite3", "local"):
        return "sqlite"
    if _sqlite_fallback_active:
        return "sqlite"
    if explicit_type in ("postgres", "postgresql", "pgsql"):
        return "postgres"
    if explicit_type == "mysql":
        return "mysql"

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
        return "postgres"
    if db_url.startswith("mysql://"):
        return "mysql"

    host = os.environ.get("DB_HOST", "").strip().lower()
    port = str(os.environ.get("DB_PORT", "")).strip()

    if "postgres" in host or host.startswith("dpg-") or port == "5432":
        return "postgres"

    return "mysql"


# ---------------------------------------------------------------------------
# Connection Factory
# ---------------------------------------------------------------------------

def get_connection():
    """Returns a database connection.

    Order: explicit config (DB_TYPE / DATABASE_URL / DB_HOST) -> local MySQL
    -> automatic SQLite fallback (backend/polyphonic.db) when no server is
    reachable, so the app always starts.
    """
    global _sqlite_fallback_active
    if get_db_type() == "sqlite":
        return _sqlite_connection()
    try:
        return _server_connection()
    except Exception as error:
        # DB_STRICT=1 turns the fallback off (fail instead of using SQLite).
        if os.environ.get("DB_STRICT", "").strip() in ("1", "true", "yes"):
            raise
        with _fallback_lock:
            if not _sqlite_fallback_active:
                where = "configured database" if _db_explicitly_configured() else "MySQL"
                print(f"[Database] {where} not reachable ({error}). Using local SQLite database at {SQLITE_PATH}")
                global _fallback_reason
                _fallback_reason = str(error)
            _sqlite_fallback_active = True
        return _sqlite_connection()


def _server_connection():
    """Returns a connected MySQL or PostgreSQL connection."""
    db_type = get_db_type()
    database_url = os.environ.get("DATABASE_URL", "").strip()

    if db_type == "postgres":
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is not installed. Run 'pip install psycopg2-binary' to enable PostgreSQL support."
            )

        if database_url:
            # Fix Render postgres:// prefix to postgresql:// if needed
            if database_url.startswith("postgres://"):
                database_url = "postgresql://" + database_url[len("postgres://"):]
            conn = psycopg2.connect(database_url, connect_timeout=5)
        else:
            conn = psycopg2.connect(
                host=os.environ.get("DB_HOST", "localhost"),
                port=int(os.environ.get("DB_PORT", 5432)),
                user=os.environ.get("DB_USER", "postgres"),
                password=os.environ.get("DB_PASSWORD", ""),
                dbname=os.environ.get("DB_NAME", "polyphonic_db"),
                connect_timeout=5,
            )
        return PostgresConnectionWrapper(conn)

    # MySQL connection
    if mysql is None:
        raise RuntimeError(
            "mysql-connector-python is not installed. Run 'pip install mysql-connector-python'."
        )

    if database_url and database_url.startswith("mysql://"):
        parsed = urlparse(database_url)
        port = parsed.port or 3306
        conn = mysql.connector.connect(
            connection_timeout=5,
            host=parsed.hostname or "localhost",
            port=port,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or "polyphonic_instrument_db",
        )
        return conn

    port_val = os.environ.get("DB_PORT", "3306")
    try:
        port = int(port_val)
    except ValueError:
        port = 3306

    connection = mysql.connector.connect(
        connection_timeout=5,
        host=os.environ.get("DB_HOST", "localhost"),
        port=port,
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "polyphonic_instrument_db"),
    )
    return connection


# ---------------------------------------------------------------------------
# Database Initialization (Auto Table Creation)
# ---------------------------------------------------------------------------

def init_database():
    """Creates all required tables if they don't already exist."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Opening a connection first lets the SQLite fallback kick in if needed.
    probe = get_connection()
    probe.close()
    db_type = get_db_type()

    if db_type == "sqlite":
        tables = get_existing_tables()
        return {
            "success": True,
            "db_type": db_type,
            "tables_found": tables,
            "total_tables": len(tables),
            "message": f"Successfully initialized {len(tables)} tables (SQLITE: {SQLITE_PATH})",
        }

    schema_file = os.path.join(
        base_dir,
        "schema_postgres.sql" if db_type == "postgres" else "schema_mysql.sql",
    )

    if not os.path.isfile(schema_file):
        schema_file = os.path.join(base_dir, "schema.sql")

    if not os.path.isfile(schema_file):
        raise FileNotFoundError(f"Schema file not found at {schema_file}")

    with open(schema_file, "r", encoding="utf-8") as f:
        sql_content = f.read()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        # Split statements by semicolon
        statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

        for stmt in statements:
            # Skip empty lines or pure comments
            lines = [line for line in stmt.splitlines() if line.strip() and not line.strip().startswith("--")]
            if not lines:
                continue
            cleaned = "\n".join(lines)
            try:
                cursor.execute(cleaned)
            except Exception as e:
                # If error is table already exists or index exists, continue safely
                err_msg = str(e).lower()
                if "already exists" in err_msg or "duplicate key" in err_msg:
                    continue
                print(f"[init_database] Warning executing statement: {e}")

        connection.commit()

        # Query existing tables to verify
        tables = get_existing_tables(connection)
        return {
            "success": True,
            "db_type": db_type,
            "tables_found": tables,
            "total_tables": len(tables),
            "message": f"Successfully initialized {len(tables)} tables ({db_type.upper()})",
        }

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass


def get_existing_tables(connection=None):
    """Returns a list of table names in the active database."""
    should_close = False
    if connection is None:
        connection = get_connection()
        should_close = True

    try:
        cursor = connection.cursor()
        db_type = get_db_type()

        if db_type == "postgres":
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            )
            rows = cursor.fetchall()
            if rows and isinstance(rows[0], dict):
                return [r["table_name"] for r in rows]
            return [r[0] for r in rows]
        elif db_type == "sqlite":
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            return [r[0] for r in cursor.fetchall()]
        else:
            cursor.execute("SHOW TABLES;")
            rows = cursor.fetchall()
            if rows and isinstance(rows[0], dict):
                return [list(r.values())[0] for r in rows]
            return [r[0] for r in rows]
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        if should_close:
            try:
                connection.close()
            except Exception:
                pass


def test_connection():
    """Tests connectivity and prints diagnostic details."""
    db_type = get_db_type()
    print(f"Testing database connection (Detected type: {db_type.upper()})...")

    try:
        connection = get_connection()
        if hasattr(connection, "is_connected") and connection.is_connected():
            print(f"[OK] {db_type.upper()} connected successfully!")
        else:
            print(f"[OK] {db_type.upper()} connection object acquired!")

        tables = get_existing_tables(connection)
        print(f"[OK] Existing tables in database ({len(tables)}): {', '.join(tables) if tables else 'None'}")

        connection.close()
        return True
    except Exception as error:
        print(f"[FAIL] {db_type.upper()} connection failed!")
        print("Error details:", error)
        return False


if __name__ == "__main__":
    test_connection()