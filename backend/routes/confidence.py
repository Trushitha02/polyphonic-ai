from flask import Blueprint, request, jsonify
from services.confidence_analysis import calculate_confidence


confidence_bp = Blueprint("confidence", __name__)


@confidence_bp.route("/calculate", methods=["POST"])
def calculate():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    performance_score = data.get("performance_score", 0)
    practice_count = data.get("practice_count", 0)

    try:

        performance_score = float(performance_score)
        practice_count = int(practice_count)

        if performance_score < 0 or performance_score > 100:
            return jsonify({
                "success": False,
                "message": "Performance score must be between 0 and 100"
            }), 400

        if practice_count < 0:
            return jsonify({
                "success": False,
                "message": "Practice count cannot be negative"
            }), 400

        confidence = calculate_confidence(
            performance_score,
            practice_count
        )

        return jsonify({
            "success": True,
            "confidence_score": confidence
        }), 200

    except (ValueError, TypeError):

        return jsonify({
            "success": False,
            "message": "Invalid performance score or practice count"
        }), 400