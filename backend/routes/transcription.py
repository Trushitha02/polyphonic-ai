from flask import Blueprint, request, jsonify
from services.transcription import transcribe_audio
from services.instrument_isolation import get_or_create_isolated_instrument


transcription_bp = Blueprint("transcription", __name__)


@transcription_bp.route("/run", methods=["POST"])
def run_transcription():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    audio_path = data.get("audio_path")
    instrument = data.get("instrument")
    audio_id = data.get("audio_id")

    if audio_id and instrument:
        try:
            isolated_path = get_or_create_isolated_instrument(
                int(audio_id),
                instrument
            )
            if isolated_path:
                audio_path = isolated_path
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "message": "Invalid audio_id"
            }), 400

    if not audio_path:
        return jsonify({
            "success": False,
            "message": "Audio path is required"
        }), 400

    try:

        result = transcribe_audio(
            audio_path,
            instrument
        )

        return jsonify({
            "success": True,
            "message": "Transcription completed",
            "result": result
        }), 200

    except Exception as error:

        return jsonify({
            "success": False,
            "message": "Transcription failed",
            "error": str(error)
        }), 500