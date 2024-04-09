import sqlite3


def get_table_sizes(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the page size
    cursor.execute("PRAGMA page_size;")
    page_size = cursor.fetchone()[0]

    # Get the list of tables and their number of pages
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Calculate and print the size of each table
    table_sizes = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        cursor.execute(f"PRAGMA table_xinfo('{table_name}');")
        data = cursor.fetchall()
        num_pages = sum([d[8] for d in data if d[8] is not None])
        table_size = num_pages * page_size
        table_sizes[table_name] = table_size

    conn.close()
    return table_sizes


# Convert size from bytes to MB and GB
def convert_size(size_bytes):
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    return size_mb, size_gb


db_path = "./data/features.db"

table_sizes = get_table_sizes(db_path)

for table_name, size in table_sizes.items():
    size_mb, size_gb = convert_size(size)
    print(f"Table '{table_name}' size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
