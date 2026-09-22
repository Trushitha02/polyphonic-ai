from flask import Blueprint, jsonify, request

from database.database import get_connection


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({
            "success": False,
            "message": "Name, email and password are required",
        }), 400

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({
                "success": False,
                "message": "Email already registered",
            }), 409

        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, password),
        )
        connection.commit()
        user_id = cursor.lastrowid

        return jsonify({
            "success": True,
            "message": "Registration successful",
            "user": {"id": user_id, "name": name, "email": email},
        }), 201
    except Exception as error:
        if connection:
            connection.rollback()
        return jsonify({
            "success": False,
            "message": "Registration failed",
            "error": str(error),
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password are required",
        }), 400

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_id, name, email, password FROM users WHERE email = %s",
            (email,),
        )
        user = cursor.fetchone()

        if not user or user["password"] != password:
            return jsonify({
                "success": False,
                "message": "Invalid email or password",
            }), 401

        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user["user_id"],
                "name": user["name"],
                "email": user["email"],
            },
        }), 200
    except Exception as error:
        return jsonify({
            "success": False,
            "message": "Login failed",
            "error": str(error),
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@auth_bp.route("/logout", methods=["POST"])
def logout():
    return jsonify({
        "success": True,
        "message": "Logout successful",
    }), 200


@auth_bp.route("/me", methods=["GET"])
def get_current_user():
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_id, name, email, created_at FROM users WHERE user_id = %s",
            (user_id,),
        )
        user = cursor.fetchone()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        user["id"] = user.pop("user_id")
        return jsonify({"success": True, "user": user}), 200
    except Exception as error:
        return jsonify({"success": False, "message": "Could not load profile", "error": str(error)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@auth_bp.route("/update-profile", methods=["POST"])
def update_profile():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")

    if not user_id or not name or not email:
        return jsonify({"success": False, "message": "User ID, name and email are required"}), 400
    if new_password and len(new_password) < 6:
        return jsonify({"success": False, "message": "New password must contain at least 6 characters"}), 400

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT password FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        if new_password and user["password"] != current_password:
            return jsonify({"success": False, "message": "Current password is incorrect"}), 401

        cursor.execute("SELECT user_id FROM users WHERE email = %s AND user_id <> %s", (email, user_id))
        if cursor.fetchone():
            return jsonify({"success": False, "message": "Email is already in use"}), 409

        if new_password:
            cursor.execute(
                "UPDATE users SET name = %s, email = %s, password = %s WHERE user_id = %s",
                (name, email, new_password, user_id),
            )
        else:
            cursor.execute(
                "UPDATE users SET name = %s, email = %s WHERE user_id = %s",
                (name, email, user_id),
            )
        connection.commit()
        return jsonify({
            "success": True,
            "message": "Profile updated successfully",
            "user": {"id": user_id, "name": name, "email": email},
        }), 200
    except Exception as error:
        if connection:
            connection.rollback()
        return jsonify({"success": False, "message": "Profile update failed", "error": str(error)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    new_password = data.get("new_password", "")

    if not email or not new_password:
        return jsonify({
            "success": False,
            "message": "Email and new password are required",
        }), 400

    if len(new_password) < 6:
        return jsonify({
            "success": False,
            "message": "Password must contain at least 6 characters",
        }), 400

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if not cursor.fetchone():
            return jsonify({
                "success": False,
                "message": "No account found for this email",
            }), 404

        cursor.execute(
            "UPDATE users SET password = %s WHERE email = %s",
            (new_password, email),
        )
        connection.commit()
        return jsonify({
            "success": True,
            "message": "Password reset successfully",
        }), 200
    except Exception as error:
        if connection:
            connection.rollback()
        return jsonify({
            "success": False,
            "message": "Password reset failed",
            "error": str(error),
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
