from flask import Blueprint, request, jsonify, current_app, send_file
from database.database import get_connection
from werkzeug.utils import secure_filename
import os


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
    user_id = request.form.get("user_id")

    # Secure filename
    filename = secure_filename(file.filename)

    # Upload folder
    upload_folder = current_app.config["UPLOAD_FOLDER"]

    os.makedirs(upload_folder, exist_ok=True)

    # Complete file path
    file_path = os.path.join(
        upload_folder,
        filename
    )

    # Save audio file
    file.save(file_path)

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
                with wave.open(file_path, "rb") as w:
                    duration = round(w.getnframes() / float(w.getframerate()), 2)
            except Exception:
                pass

        return jsonify({

            "success": True,

            "message": "Audio uploaded successfully",

            "audio_id": audio_id,

            "filename": filename,

            "file_path": file_path,

            "duration": duration,

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