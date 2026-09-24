from flask import Blueprint, request, jsonify
from services.performance_analysis import analyze_performance
from database.database import get_connection
from services.instrument_isolation import get_or_create_isolated_instrument
from services.instrument_separation import scan_existing_stems
import os


performance_bp = Blueprint("performance", __name__)


def get_audio_path(audio_id):
    if not audio_id:
        return None
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT file_path FROM audio_files WHERE audio_id = %s", (audio_id,))
        row = cursor.fetchone()
        return row.get("file_path") if row else None
    except Exception:
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@performance_bp.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    reference_audio = data.get("reference_audio")
    performance_audio = data.get("performance_audio")
    user_id = data.get("user_id")
    audio_id = data.get("audio_id")
    instrument = data.get("instrument")

    if audio_id and instrument == "Vocals":
        base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "separated", str(audio_id)))
        stems = scan_existing_stems(base_folder)
        reference_audio = stems.get("vocals") or reference_audio
    elif audio_id and instrument and instrument != "BGM":
        isolated_path = get_or_create_isolated_instrument(audio_id, instrument)
        if isolated_path:
            reference_audio = isolated_path
    elif audio_id and instrument == "BGM":
        base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "separated", str(audio_id)))
        stems = scan_existing_stems(base_folder)
        reference_audio = stems.get("bgm") or stems.get("other") or reference_audio

    if (not reference_audio or not os.path.isfile(reference_audio)) and audio_id:
        database_path = get_audio_path(audio_id)
        if database_path and os.path.isfile(database_path):
            reference_audio = database_path

    if not reference_audio or not performance_audio:
        return jsonify({
            "success": False,
            "message": "Both reference and performance audio are required"
        }), 400

    try:

        # Analyze the user's performance
        result = analyze_performance(
            reference_audio,
            performance_audio
        )

        # Save performance result to MySQL
        if user_id:

            connection = get_connection()
            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO performances
                (
                    user_id,
                    audio_id,
                    instrument_name,
                    pitch_score,
                    rhythm_score,
                    timing_score,
                    note_score,
                    overall_score
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    audio_id,
                    instrument,
                    result.get("pitch_score", 0),
                    result.get("rhythm_score", 0),
                    result.get("timing_score", 0),
                    result.get("note_score", 0),
                    result.get("overall_score", 0)
                )
            )

            connection.commit()

            cursor.close()
            connection.close()

        return jsonify({

            "success": True,

            "message":
                "Performance analyzed successfully",

            "result": result

        }), 200

    except Exception as error:

        return jsonify({

            "success": False,

            "message":
                "Performance analysis failed",

            "error":
                str(error)

        }), 500