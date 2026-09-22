from flask import Blueprint, jsonify, request
from database.database import get_connection


progress_bp = Blueprint("progress", __name__)


def _level_for_score(score):
    if score >= 96:
        return "Master Performer"
    if score >= 90:
        return "Advanced"
    if score >= 75:
        return "Intermediate"
    if score >= 60:
        return "Developing Musician"
    if score >= 40:
        return "Beginner"
    return "Musical Starter"


@progress_bp.route("/dashboard", methods=["GET"])
def get_dashboard():
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_performances,
                COUNT(DISTINCT audio_id) AS total_songs,
                COALESCE(MAX(overall_score), 0) AS best_score,
                COALESCE(AVG(overall_score), 0) AS average_score,
                COALESCE(AVG(pitch_score), 0) AS pitch,
                COALESCE(AVG(rhythm_score), 0) AS rhythm,
                COALESCE(AVG(timing_score), 0) AS timing,
                COALESCE(AVG(note_score), 0) AS notes
            FROM performances
            WHERE user_id = %s
            """,
            (user_id,),
        )
        stats = cursor.fetchone() or {}
        cursor.execute(
            """
            SELECT performance_id, audio_id, instrument_name,
                   pitch_score, rhythm_score, timing_score,
                   note_score, overall_score, created_at
            FROM performances
            WHERE user_id = %s
            ORDER BY created_at DESC, performance_id DESC
            LIMIT 20
            """,
            (user_id,),
        )
        recent = cursor.fetchall()

        def number(value):
            return round(float(value or 0), 2)

        latest = recent[0] if recent else {}
        latest_score = number(latest.get("overall_score"))
        average_score = number(stats.get("average_score"))
        response = {
            "success": True,
            "stats": {
                "total_performances": int(stats.get("total_performances") or 0),
                "total_songs": int(stats.get("total_songs") or 0),
                "best_score": number(stats.get("best_score")),
                "latest_score": latest_score,
                "average_score": average_score,
            },
            "skill_scores": {
                "pitch": number(stats.get("pitch")),
                "rhythm": number(stats.get("rhythm")),
                "timing": number(stats.get("timing")),
                "notes": number(stats.get("notes")),
            },
            "level": _level_for_score(latest_score or average_score),
            "recent_performances": recent,
        }
        return jsonify(response), 200
    except Exception as error:
        return jsonify({"success": False, "message": "Could not load performance history", "error": str(error)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@progress_bp.route("/<int:user_id>", methods=["GET"])
def get_progress(user_id):

    try:

        connection = get_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT *
            FROM progress
            WHERE user_id = %s
            """,
            (user_id,)
        )

        progress = cursor.fetchone()

        cursor.close()
        connection.close()

        # No progress data yet
        if not progress:

            return jsonify({
                "success": True,
                "message": "No progress data yet",
                "progress": {
                    "overall_score": 0,
                    "confidence_score": 0,
                    "songs_completed": 0,
                    "performances_count": 0,
                    "current_level": "Musical Starter"
                }
            }), 200

        return jsonify({
            "success": True,
            "progress": progress
        }), 200

    except Exception as error:

        return jsonify({
            "success": False,
            "message": "Could not fetch progress",
            "error": str(error)
        }), 500