from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

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
SEPARATED_FOLDER = os.path.join(BASE_DIR, "separated_audio")
MIDI_FOLDER = os.path.join(BASE_DIR, "midi")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/api/test")
def test():
    return jsonify({
        "message": "Backend connection successful!",
        "project": "Polyphonic Instrument Separation and Transcription"
    })


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )