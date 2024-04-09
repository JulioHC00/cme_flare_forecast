import sqlite3


def vacuum_database(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Vacuum the database
    cursor.execute("VACUUM;")

    conn.close()


if __name__ == "__main__":
    # Replace 'data/features.db' with your database file path
    db_path = "./data/features.db"
    vacuum_database(db_path)
