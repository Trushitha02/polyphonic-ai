from flask import Blueprint, request, jsonify
from services.level_prediction import predict_level


level_bp = Blueprint("level", __name__)


@level_bp.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    score = data.get("score")

    if score is None:
        return jsonify({
            "success": False,
            "message": "Score is required"
        }), 400

    try:

        score = float(score)

        # Keep score within 0–100
        if score < 0 or score > 100:
            return jsonify({
                "success": False,
                "message": "Score must be between 0 and 100"
            }), 400

        level = predict_level(score)

        return jsonify({
            "success": True,
            "score": score,
            "level": level
        }), 200

    except ValueError:

        return jsonify({
            "success": False,
            "message": "Score must be a number"
        }), 400