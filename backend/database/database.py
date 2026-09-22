import mysql.connector
import os


def get_connection():
    connection = mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "polyphonic_instrument_db"),
    )

    return connection


def test_connection():

    try:
        connection = get_connection()

        if connection.is_connected():
            print("MySQL connected successfully!")

        connection.close()

    except mysql.connector.Error as error:
        print("MySQL connection failed!")
        print("Error:", error)


if __name__ == "__main__":
    test_connection()