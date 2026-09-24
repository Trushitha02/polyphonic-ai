from flask import Blueprint, request, jsonify, current_app, send_file
from database.database import get_connection
from werkzeug.utils import secure_filename
import os
import uuid

from services.instrument_detection import (
    detect_audio_instruments,
    save_instrument_detections,
    get_saved_instrument_detections
)


audio_bp = Blueprint("audio", __name__)


@audio_bp.route("/file", methods=["GET"])
def get_uploaded_audio():
    upload_folder = os.path.abspath(current_app.config["UPLOAD_FOLDER"])
    dataset_folder = os.path.abspath(current_app.config.get("DATASET_FOLDER", ""))
    path_param = request.args.get("path", "").strip()
    audio_id = request.args.get("audio_id")

    requested_path = None

    if audio_id:
        try:
            conn = get_connection()
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT file_path FROM audio_files WHERE audio_id = %s", (audio_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row and row.get("file_path"):
                requested_path = os.path.abspath(row["file_path"])
        except Exception as e:
            print("DB error fetching audio path:", e)

    if not requested_path and path_param:
        requested_path = os.path.abspath(path_param)

    if not requested_path:
        return jsonify({"success": False, "message": "No audio path or audio_id provided"}), 400

    # Verify the path is within permitted project directories
    valid_prefix = False
    for allowed_dir in [upload_folder, dataset_folder]:
        if allowed_dir and os.path.normcase(requested_path).startswith(os.path.normcase(allowed_dir) + os.sep):
            valid_prefix = True
            break

    if not valid_prefix:
        # Check if file exists within project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if os.path.normcase(requested_path).startswith(os.path.normcase(project_root) + os.sep):
            valid_prefix = True

    if not valid_prefix:
        return jsonify({"success": False, "message": "Invalid audio path"}), 400

    if os.path.splitext(requested_path)[1].lower().lstrip(".") not in ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "message": "Only audio files can be streamed"}), 400

    if not os.path.isfile(requested_path):
        return jsonify({"success": False, "message": "Audio file not found"}), 404

    lower_ext = os.path.splitext(requested_path)[1].lower()
    mimetype = "audio/mpeg" if lower_ext == ".mp3" else ("audio/wav" if lower_ext == ".wav" else None)

    response = send_file(
        requested_path,
        as_attachment=request.args.get("download") == "1",
        download_name=os.path.basename(requested_path),
        conditional=True,
        mimetype=mimetype,
    )
    response.headers["Accept-Ranges"] = "bytes"
    return response


# Allowed audio formats
ALLOWED_EXTENSIONS = {
    "mp3",
    "wav",
    "flac",
    "ogg",
    "m4a"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------- UPLOAD AUDIO ----------------
@audio_bp.route("/upload", methods=["POST"])
def upload_audio():

    # Check whether file exists
    if "file" not in request.files:

        return jsonify({
            "success": False,
            "message": "No audio file uploaded"
        }), 400

    file = request.files["file"]

    # Check filename
    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "No file selected"
        }), 400

    # Check file format
    if not allowed_file(file.filename):

        return jsonify({
            "success": False,
            "message": "Unsupported audio format"
        }), 400

    # Get user ID
    user_id = request.form.get("user_id") or None
    purpose = (request.form.get("purpose") or "").strip().lower()

    # Secure filename (keep a readable name even for non-latin titles)
    filename = secure_filename(file.filename)
    extension = file.filename.rsplit(".", 1)[1].lower()
    if not filename or "." not in filename:
        filename = f"audio_{uuid.uuid4().hex[:8]}.{extension}"

    # Upload folder
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    if purpose == "performance":
        upload_folder = os.path.join(upload_folder, "performances")

    os.makedirs(upload_folder, exist_ok=True)

    # Complete file path. Re-uploading the same song reuses its file;
    # a different song with the same name gets a unique name instead of
    # overwriting (which would break the earlier song's separation).
    file_path = os.path.join(upload_folder, filename)
    if os.path.exists(file_path):
        temp_path = os.path.join(upload_folder, f".incoming_{uuid.uuid4().hex}")
        file.save(temp_path)
        if os.path.getsize(temp_path) == os.path.getsize(file_path):
            os.replace(temp_path, file_path)
        else:
            stem, ext = os.path.splitext(filename)
            filename = f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
            file_path = os.path.join(upload_folder, filename)
            os.replace(temp_path, file_path)
    else:
        file.save(file_path)

    # A practice recording for Performance Analysis is only compared with
    # the reference; it is not a new song, so skip DB + instrument detection.
    if purpose == "performance":
        return jsonify({
            "success": True,
            "message": "Performance audio uploaded",
            "filename": filename,
            "file_path": file_path,
        }), 201

    try:

        connection = get_connection()

        cursor = connection.cursor()

        # Insert information into MySQL
        cursor.execute(
            """
            INSERT INTO audio_files
            (user_id, original_name, file_path)
            VALUES (%s, %s, %s)
            """,
            (
                user_id,
                filename,
                file_path
            )
        )

        connection.commit()

        audio_id = cursor.lastrowid

        cursor.close()
        connection.close()

        # Compute file duration
        duration = None
        try:
            import soundfile as sf
            info = sf.info(file_path)
            duration = round(float(info.duration), 2)
        except Exception:
            try:
                import wave
                with wave.open(file_path, "rb") as audio_wave:
                    duration = round(
                        audio_wave.getnframes() / float(audio_wave.getframerate()),
                        2,
                    )
            except Exception:
                pass

        # Run AI instrument detection on uploaded audio
        detected_instruments = []
        try:
            detected_instruments = detect_audio_instruments(file_path, source="Uploaded Audio")
            if detected_instruments:
                save_instrument_detections(audio_id, detected_instruments)
        except Exception as det_err:
            print(f"[Upload] Instrument detection warning for audio_id {audio_id}: {det_err}")

        return jsonify({
            "success": True,
            "message": "Audio uploaded and analyzed successfully",
            "audio_id": audio_id,
            "filename": filename,
            "file_path": file_path,
            "duration": duration,
            "detected_instruments": detected_instruments,
            "detected_instrument_count": len(detected_instruments),
        }), 201

    except Exception as error:

        return jsonify({

            "success": False,

            "message": "Database error",

            "error": str(error)

        }), 500


# ---------------- GET AUDIO ----------------
@audio_bp.route("/<int:audio_id>", methods=["GET"])
def get_audio(audio_id):

    try:

        connection = get_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT *
            FROM audio_files
            WHERE audio_id = %s
            """,
            (audio_id,)
        )

        audio = cursor.fetchone()

        cursor.close()
        connection.close()

        if not audio:

            return jsonify({
                "success": False,
                "message": "Audio not found"
            }), 404

        # Compute audio duration
        duration = None
        file_path = audio.get("file_path")
        if file_path and os.path.isfile(file_path):
            try:
                import soundfile as sf
                info = sf.info(file_path)
                duration = round(float(info.duration), 2)
            except Exception:
                try:
                    import wave
                    with wave.open(file_path, "rb") as w:
                        duration = round(w.getnframes() / float(w.getframerate()), 2)
                except Exception:
                    pass
        audio["duration"] = duration

        # Include saved AI detected instruments
        try:
            detected_instruments = get_saved_instrument_detections(audio_id)
            if not detected_instruments and file_path and os.path.isfile(file_path):
                detected_instruments = detect_audio_instruments(file_path, source="Uploaded Audio")
                if detected_instruments:
                    save_instrument_detections(audio_id, detected_instruments)
            audio["detected_instruments"] = detected_instruments
            audio["detected_instrument_count"] = len(detected_instruments)
        except Exception as det_err:
            print(f"[Get Audio] Instrument detection fetch error for audio_id {audio_id}: {det_err}")
            audio["detected_instruments"] = []
            audio["detected_instrument_count"] = 0

        return jsonify({

            "success": True,

            "audio": audio

        }), 200

    except Exception as error:

        return jsonify({

            "success": False,

            "message": "Could not fetch audio",

            "error": str(error)

        }), 500