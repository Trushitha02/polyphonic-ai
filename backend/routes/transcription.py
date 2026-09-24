import json
import os

from flask import Blueprint, current_app, jsonify, request, send_file, url_for

from database.database import get_connection
from services.instrument_isolation import get_or_create_isolated_instrument
from services.transcription import transcribe_audio


transcription_bp = Blueprint("transcription", __name__)


def _safe_name(value):
    return "".join(c if c.isalnum() else "_" for c in (value or "audio")).strip("_") or "audio"


def _load_cached(audio_id, instrument, audio_path):
    """Return a saved transcription if the source audio has not changed since."""
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT transcription_id, midi_path, notes_data
            FROM transcriptions
            WHERE audio_id = %s AND instrument_name = %s
            ORDER BY transcription_id DESC
            LIMIT 1
            """,
            (audio_id, instrument),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or not row.get("notes_data"):
            return None
        midi_path = row.get("midi_path")
        if not midi_path or not os.path.isfile(midi_path):
            return None
        if os.path.getmtime(midi_path) < os.path.getmtime(audio_path):
            return None
        result = json.loads(row["notes_data"])
        if result.get("audio_path") != os.path.abspath(audio_path):
            return None
        return result
    except Exception as error:
        print("Transcription cache lookup warning:", error)
        return None


def _save(audio_id, instrument, midi_path, result):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM transcriptions WHERE audio_id = %s AND instrument_name = %s",
            (audio_id, instrument),
        )
        cur.execute(
            """
            INSERT INTO transcriptions (audio_id, instrument_name, midi_path, notes_data)
            VALUES (%s, %s, %s, %s)
            """,
            (audio_id, instrument, midi_path, json.dumps(result)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as error:
        print("Could not save transcription:", error)


@transcription_bp.route("/run", methods=["POST"])
def run_transcription():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    audio_path = data.get("audio_path")
    instrument = (data.get("instrument") or "").strip() or None
    audio_id = data.get("audio_id")

    if audio_id:
        try:
            audio_id = int(audio_id)
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "message": "Invalid audio_id"
            }), 400

    if audio_id and instrument:
        isolated_path = get_or_create_isolated_instrument(audio_id, instrument)
        if isolated_path:
            audio_path = isolated_path

    if not audio_path:
        return jsonify({
            "success": False,
            "message": "Audio path is required"
        }), 400

    if not os.path.isfile(audio_path):
        return jsonify({
            "success": False,
            "message": "The audio for this instrument was not found. Run AI separation first."
        }), 404

    try:
        result = None
        if audio_id and instrument:
            result = _load_cached(audio_id, instrument, audio_path)

        if result is None:
            midi_folder = current_app.config.get("MIDI_FOLDER")
            midi_name = f"{audio_id or 'audio'}_{_safe_name(instrument)}.mid"
            midi_path = os.path.join(midi_folder, midi_name)
            result = transcribe_audio(audio_path, instrument, midi_path=midi_path)
            if audio_id and instrument:
                _save(audio_id, instrument, midi_path, result)

        midi_file = os.path.basename(result.get("midi_path") or "")
        if midi_file:
            result["midi_url"] = url_for("transcription.download_midi", name=midi_file)

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


@transcription_bp.route("/midi/<name>", methods=["GET"])
def download_midi(name):
    midi_folder = os.path.abspath(current_app.config.get("MIDI_FOLDER"))
    path = os.path.abspath(os.path.join(midi_folder, name))
    if not path.startswith(midi_folder + os.sep) or not path.lower().endswith(".mid"):
        return jsonify({"success": False, "message": "Invalid MIDI file"}), 400
    if not os.path.isfile(path):
        return jsonify({"success": False, "message": "MIDI file not found"}), 404
    return send_file(path, as_attachment=True, download_name=name, mimetype="audio/midi")


@transcription_bp.route("/history/<int:audio_id>", methods=["GET"])
def transcription_history(audio_id):
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT transcription_id, instrument_name, midi_path, created_at
            FROM transcriptions WHERE audio_id = %s ORDER BY transcription_id DESC
            """,
            (audio_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        items = []
        for row in rows:
            name = os.path.basename(row.get("midi_path") or "")
            items.append({
                "transcription_id": row["transcription_id"],
                "instrument": row["instrument_name"],
                "midi_url": url_for("transcription.download_midi", name=name) if name else None,
                "created_at": str(row["created_at"]) if row.get("created_at") else None,
            })
        return jsonify({"success": True, "transcriptions": items}), 200
    except Exception as error:
        return jsonify({"success": False, "message": "Could not load transcriptions", "error": str(error)}), 500
