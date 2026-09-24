"""
Polyphonic AI - Database Initialization Script
Initializes all 8 tables for Render deployment (PostgreSQL or MySQL).
Can be run locally or inside Render's web shell / build process.

Usage examples:
    # Use environment variables:
    python backend/database/init_db.py

    # Point to a specific database URL (e.g., Render PostgreSQL):
    python backend/database/init_db.py --url "postgres://user:password@host.render.com/dbname"

    # Point to an external MySQL instance:
    python backend/database/init_db.py --host db.example.com --user root --password mypass --database polyphonic_db --port 3306
"""

import argparse
import os
import sys

# Ensure backend root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from database.database import (
    get_connection,
    get_db_type,
    get_existing_tables,
    init_database,
)

EXPECTED_TABLES = [
    "users",
    "audio_files",
    "performances",
    "progress",
    "instrument_detections",
    "separation_results",
    "practice_recommendations",
    "transcriptions",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize Polyphonic Database Tables")
    parser.add_argument("--url", help="Database connection URL (Postgres or MySQL)")
    parser.add_argument("--host", help="Database Host")
    parser.add_argument("--port", help="Database Port")
    parser.add_argument("--user", help="Database Username")
    parser.add_argument("--password", help="Database Password")
    parser.add_argument("--database", help="Database Name")
    parser.add_argument("--type", choices=["postgres", "mysql"], help="Explicit database type")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides if provided
    if args.url:
        os.environ["DATABASE_URL"] = args.url
    if args.host:
        os.environ["DB_HOST"] = args.host
    if args.port:
        os.environ["DB_PORT"] = args.port
    if args.user:
        os.environ["DB_USER"] = args.user
    if args.password:
        os.environ["DB_PASSWORD"] = args.password
    if args.database:
        os.environ["DB_NAME"] = args.database
    if args.type:
        os.environ["DB_TYPE"] = args.type

    print("=" * 60)
    print(" Polyphonic AI - Database Table Initializer")
    print("=" * 60)

    db_type = get_db_type()
    print(f"[+] Target Database Engine: {db_type.upper()}")

    host = os.environ.get("DB_HOST", "localhost")
    db_name = os.environ.get("DB_NAME", "default")
    if os.environ.get("DATABASE_URL"):
        masked = os.environ["DATABASE_URL"].split("@")[-1] if "@" in os.environ["DATABASE_URL"] else "provided URL"
        print(f"[+] Connecting via URL to: {masked}")
    else:
        print(f"[+] Connecting to Host: {host} | Database: {db_name}")

    print("\n[+] Creating tables if they do not exist...")
    try:
        result = init_database()
        print(f"[OK] {result['message']}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        sys.exit(1)

    # Verification
    print("\n[+] Verifying tables:")
    try:
        conn = get_connection()
        cur = conn.cursor()
        existing = get_existing_tables(conn)

        for table in EXPECTED_TABLES:
            if table in existing:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cur.fetchone()
                    if isinstance(count, dict):
                        count = next(iter(count.values()))
                    elif isinstance(count, (tuple, list)):
                        count = count[0]
                    print(f"  [OK] {table:<25} (rows: {count})")
                except Exception as query_err:
                    print(f"  [OK] {table:<25}")
            else:
                print(f"  [MISSING] {table:<25}")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"[WARNING] Verification error: {e}")

    missing = [t for t in EXPECTED_TABLES if t not in existing]
    if missing:
        print(f"\n[!] Warning: Missing tables: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All 8 tables are present and ready for Render backend!")
        print("          You can now run and access the backend on your phone without database errors.")


if __name__ == "__main__":
    main()
