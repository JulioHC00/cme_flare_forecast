import pandas as pd
import sqlite3
import os
from tqdm import tqdm

sharps_db_path = "data/features.db"
sharps_csv_base = "../../cmesrc/data/raw/mvts/SWAN/"


sharps_db_conn = sqlite3.connect(sharps_db_path)
sharps_db_cursor = sharps_db_conn.cursor()

# Drop the table if it exists
table_name = "SWAN"

sharps_db_cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

# Now need to walk all the directories in sharps_csv_base and load all the files

for root, dirs, files in os.walk(sharps_csv_base):
    for file in tqdm(files):
        if file.endswith(".csv"):
            # harpnum is the filename without the csv
            harpnum = int(file[:-4])

            file_path = os.path.join(root, file)
            sharps_df = pd.read_csv(os.path.join(root, file), sep="\t")
            sharps_df["harpnum"] = harpnum

            # Reorder columns to have harpnum first
            sharps_df = sharps_df[["harpnum"] + sharps_df.columns[:-1].tolist()]

            # Sort to make sure the timestamps are in order
            sharps_df = sharps_df.sort_values(by=["Timestamp"], ascending=True)

            # Check if table exists
            sharps_df.to_sql(
                table_name, sharps_db_conn, if_exists="append", index=False
            )


sharps_db_conn.commit()
sharps_db_conn.close()
