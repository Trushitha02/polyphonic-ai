from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os


def load_env_file(path):
    """Load KEY=VALUE lines from backend/.env (keeps passwords out of code)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from routes.auth import auth_bp
from routes.audio import audio_bp
from routes.separation import separation_bp
from routes.transcription import transcription_bp
from routes.performance import performance_bp
from routes.level import level_bp
from routes.confidence import confidence_bp
from routes.progress import progress_bp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
)

CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATASET_FOLDER = os.path.join(BASE_DIR, "datasets")
SEPARATED_FOLDER = os.path.join(BASE_DIR, "separated")
MIDI_FOLDER = os.path.join(BASE_DIR, "midi")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(SEPARATED_FOLDER, exist_ok=True)
os.makedirs(MIDI_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DATASET_FOLDER"] = DATASET_FOLDER
app.config["SEPARATED_FOLDER"] = SEPARATED_FOLDER
app.config["MIDI_FOLDER"] = MIDI_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER


# Register API routes
app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(audio_bp, url_prefix="/api/audio")
app.register_blueprint(separation_bp, url_prefix="/api/separation")
app.register_blueprint(transcription_bp, url_prefix="/api/transcription")
app.register_blueprint(performance_bp, url_prefix="/api/performance")
app.register_blueprint(level_bp, url_prefix="/api/level")
app.register_blueprint(confidence_bp, url_prefix="/api/confidence")
app.register_blueprint(progress_bp, url_prefix="/api/progress")

# Auto-initialize database tables on app startup
try:
    from database.database import init_database
    init_res = init_database()
    print(f"[Polyphonic Backend] Database auto-initialized: {init_res.get('message')}")
except Exception as _db_err:
    print(f"[Polyphonic Backend] Database initialization deferred/warning: {_db_err}")


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/api/test")
def test():
    return jsonify({
        "message": "Backend connection successful!",
        "project": "Polyphonic Instrument Separation and Transcription"
    })


@app.route("/api/database/status")
def database_status():
    try:
        from database.database import get_db_type, get_existing_tables, get_connection, get_fallback_reason
        conn = get_connection()
        db_type = get_db_type()
        tables = get_existing_tables(conn)
        conn.close()
        return jsonify({
            "success": True,
            "db_type": db_type,
            "fallback_reason": get_fallback_reason(),
            "tables": tables,
            "total_tables": len(tables),
            "status": "ready" if len(tables) >= 6 else "incomplete",
        }), 200
    except Exception as error:
        return jsonify({
            "success": False,
            "message": "Database connection failed",
            "error": str(error),
        }), 500


@app.route("/api/database/init", methods=["GET", "POST"])
def database_init():
    try:
        from database.database import init_database
        res = init_database()
        return jsonify(res), 200
    except Exception as error:
        return jsonify({
            "success": False,
            "message": "Database initialization failed",
            "error": str(error),
        }), 500


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )