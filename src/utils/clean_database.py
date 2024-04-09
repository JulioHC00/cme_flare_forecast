import sqlite3

db_path = "./data/features.db"

# Connect

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get all table names

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
tables = [table[0] for table in tables]

# Now drop temp tables

for table in tables:
    if table.startswith("temp_"):
        print(f"Dropping {table}")
        cur.execute(f"DROP TABLE {table}")

# Ask if this is ok
is_ok = input("Is this ok? (y/n) ")

if is_ok == "y":
    conn.commit()
    print("Done")
else:
    print("Aborting")