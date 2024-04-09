import sqlite3
import numpy as np
from datetime import datetime
from typing import Tuple
from tqdm import tqdm

db_path = "data/features.db"

cmesrc_db_path = "data/cmesrc.db"

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS keywords_desc")

# First let's create a description table
create_desc_table_sql = """
CREATE TABLE IF NOT EXISTS keywords_desc (
    keyword_name TEXT PRIMARY KEY,
    description TEXT,
    category TEXT
)
"""

# Execute the create table command
cur.execute(create_desc_table_sql)
conn.commit()

# Template to insert data into the table
insert_sql = """
INSERT INTO keywords_desc (keyword_name, description, category) VALUES (?, ?, ?)
"""

# Define the parameters and descriptions as seen in the image
keywords_desc = [
    ("ABSNJZH", "Absolute value of the net current helicity in G2/m", "SHARP"),
    ("EPSX", "Sum of X-component of normalized Lorentz force", "SWAN-SF"),
    ("EPSY", "Sum of Y-component of normalized Lorentz force", "SWAN-SF"),
    ("EPSZ", "Sum of Z-component of normalized Lorentz force", "SWAN-SF"),
    ("MEANALP", "Mean twist parameter", "SHARP"),
    ("MEANGAM", "Mean inclination angle", "SHARP"),
    ("MEANGBH", "Mean value of the horizontal field gradient", "SHARP"),
    ("MEANGBT", "Mean value of the total field gradient", "SHARP"),
    ("MEANGBZ", "Mean value of the vertical field gradient", "SHARP"),
    ("MEANJZD", "Mean vertical current density", "SHARP"),
    ("MEANJZH", "Mean current helicity", "SHARP"),
    ("MEANPOT", "Mean photospheric excess magnetic energy density", "SHARP"),
    ("MEANSHR", "Mean shear angle", "SHARP"),
    (
        "R_VALUE",
        "Total unsigned flux around high gradient polarity inversion lines using the Blos component",
        "SWAN-SF",
    ),
    ("SAVNCPP", "Sum of the absolute value of the net current per polarity", "SHARP"),
    ("SHRGT45", "Area with shear angle greater than 45 degrees", "SHARP"),
    ("TOTBSQ", "Total magnitude of Lorentz force", "SWAN-SF"),
    ("TOTFX", "Sum of X-component of Lorentz force", "SWAN-SF"),
    ("TOTFY", "Sum of Y-component of Lorentz force", "SWAN-SF"),
    ("TOTFZ", "Sum of Z-component of Lorentz force", "SWAN-SF"),
    ("TOTPOT", "Total photospheric magnetic energy density", "SHARP"),
    ("TOTUSJH", "Total unsigned current helicity", "SHARP"),
    ("TOTUSJZ", "Total unsigned vertical current", "SHARP"),
    ("USFLUX", "Total unsigned flux in Maxwells", "SHARP"),
]

keywords = [x[0] for x in keywords_desc]

other_features = ["harpnum", "Timestamp", "IS_TMFI"]

# Insert each parameter into the database
for keyword_desc in keywords_desc:
    cur.execute(insert_sql, keyword_desc)

# Now create the table for the padded features
cur.execute("DROP TABLE IF EXISTS padded_features")

# Select other_features + keywords
cur.execute(
    f"""CREATE TABLE padded_features AS
SELECT {','.join(other_features)}, {','.join(keywords)} FROM SWAN
"""
)

# Now we need to create a column IS_VALID which tells us whether an entry is ok for the model
cur.execute("ALTER TABLE padded_features ADD COLUMN IS_VALID INTEGER")

# IS_TMFI AND not missing any keywords (not null)
cur.execute(
    f"""
    UPDATE padded_features
    SET IS_VALID = (
        SELECT (
            IS_TMFI AND 
            {" AND ".join([f"{keyword} IS NOT NULL" for keyword in keywords])}
            )
            )
    """
)

cur.execute(
    "CREATE INDEX IF NOT EXISTS padded_features_harpnum ON padded_features(harpnum)"
)
cur.execute(
    "CREATE INDEX IF NOT EXISTS padded_features_timestamp ON padded_features(Timestamp)"
)
cur.execute(
    "CREATE INDEX IF NOT EXISTS padded_features_harpnum_timestamp ON padded_features(harpnum, Timestamp)"
)


# Additionally we want to add some extra features related to the flare and CME history
# These are as defined in Liu et al. 2019

# We need to load the flare list by connecting to the cmesrc db
cmesrc = sqlite3.connect(cmesrc_db_path)

# We do this harpnum by harpnum

harpnums = conn.execute("SELECT DISTINCT harpnum FROM padded_features").fetchall()
harpnums = [x[0] for x in harpnums]


def calculate_history_params(
    diffs: np.ndarray,
    tau: float,
    intensity_multipliers: np.ndarray,
    log_intensity_multipliers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the history parameter for a given tau
    """
    base = np.exp(-diffs / tau)

    hist_params = np.sum(base, axis=1)
    intensity_params = np.sum(base * intensity_multipliers, axis=1)
    log_intensity_params = np.sum(base * log_intensity_multipliers, axis=1)

    return (hist_params, intensity_params, log_intensity_params)


def solar_flare_intensity(flare_class):
    """
    Convert solar flare class to a numerical intensity value.

    Parameters:
    flare_class (str): The class of the solar flare (e.g., 'M1', 'X5').

    Returns:
    float: The numerical intensity value of the solar flare.
    """

    # Base multipliers for each class
    base_multipliers = {"A": 1, "B": 10, "C": 100, "M": 1000, "X": 10000}

    # Extract the class letter and subclass number from the flare class
    class_letter = flare_class[0]
    subclass_number = float(flare_class[1:])

    # Calculate the intensity
    intensity = base_multipliers[class_letter] * subclass_number

    return intensity


FLARE_CLASSES = ["B", "C", "M", "X"]

for flare_class in FLARE_CLASSES:
    # Add a coumn for each flare class
    cur.execute(f"ALTER TABLE padded_features ADD COLUMN {flare_class}dec REAL")
    cur.execute(f"ALTER TABLE padded_features ADD COLUMN {flare_class}his int")
    cur.execute(f"ALTER TABLE padded_features ADD COLUMN {flare_class}his1d int")

INTENSITY_CLASSES = ["E", "logE"]

for intensity_class in INTENSITY_CLASSES:
    cur.execute(f"ALTER TABLE padded_features ADD COLUMN {intensity_class}dec REAL")

for harpnum in tqdm(harpnums, desc="Calculating flare history parameters"):
    # We'll now do flare class by flare class
    overall_intensity_params = np.array([])
    overall_log_intensity_params = np.array([])

    # Now we need need all the timestamps for the harpnum

    str_harpnum_timestamps = conn.execute(
        f"""
        SELECT padded_features.Timestamp
        FROM padded_features
        WHERE padded_features.harpnum = {harpnum}
        """
    ).fetchall()

    str_harpnum_timestamps = [x[0] for x in str_harpnum_timestamps]

    # And do the same thing
    harpnum_timestamps = np.array(
        [x for x in str_harpnum_timestamps], dtype="datetime64[s]"
    )[:, np.newaxis]

    for flare_class in ["B", "C", "M", "X"]:
        # We'll do this by first selecting all the flares of that class
        flares = cmesrc.execute(
            f"""
            SELECT 
            flare_date, flare_class
            FROM flares
            WHERE harpnum = {harpnum} AND flare_class LIKE '{flare_class}%'
            """
        ).fetchall()

        # Now put the timestamps into a numpy array as numpy datetime64

        timestamps = np.array([x[0] for x in flares], dtype="datetime64[s]")

        flare_classes = np.array([x[1] for x in flares])

        flare_intensities = np.array([solar_flare_intensity(x) for x in flare_classes])
        log_flare_intensities = np.log10(flare_intensities)

        # Now calculate diffs which is timestamp - flare_time. In the end we should have a matrix of shape (n_harpnum_timestamps, n_flares)
        # the diffs must be in hours
        # And we use numpy for efficiency
        diffs = (harpnum_timestamps - timestamps) / np.timedelta64(1, "h")

        # Now we extract the N flares history parameters before replacing negative values with np.inf

        his_params = np.sum((diffs > 0).astype(int), axis=1).astype(int).tolist()
        his1d_params = (
            np.sum(((diffs < 24) & (diffs > 0)).astype(int), axis=1)
            .astype(int)
            .tolist()
        )

        # Need to replace negative values with np.inf
        diffs = np.where(diffs < 0, np.inf, diffs)

        # For the intensities and log intensities need to repeat them for each timestamp
        # So shape becomes (n_harpnum_timestamps, n_flares)

        flare_intensities = np.repeat(flare_intensities[np.newaxis, :], len(diffs), 0)
        log_flare_intensities = np.repeat(
            log_flare_intensities[np.newaxis, :], len(diffs), 0
        )

        (
            history_params,
            intensity_params,
            log_intensity_params,
        ) = calculate_history_params(
            diffs, 12, flare_intensities, log_flare_intensities
        )

        if len(overall_intensity_params) == 0:
            overall_intensity_params = intensity_params
            overall_log_intensity_params = log_intensity_params
        else:
            overall_intensity_params += intensity_params
            overall_log_intensity_params += log_intensity_params

        # Now update the database
        # Keep in mind history_params has shape (n_harpnum_timestamps) and we need to update the database for each timestamp
        cur.executemany(
            f"""
            UPDATE padded_features
            SET {flare_class}dec = ?, {flare_class}his = ?, {flare_class}his1d = ?
            WHERE harpnum = ? AND Timestamp = ?
            """,
            zip(
                history_params,
                his_params,
                his1d_params,
                [harpnum] * len(history_params),
                str_harpnum_timestamps,
            ),
        )

    # Now update with the overall intensity params

    cur.executemany(
        """
        UPDATE padded_features
        SET Edec = ?, logEdec = ?
        WHERE harpnum = ? AND Timestamp = ?
    """,
        zip(
            overall_intensity_params,
            overall_log_intensity_params,
            [harpnum] * len(str_harpnum_timestamps),
            str_harpnum_timestamps,
        ),
    )

cur.execute("ALTER TABLE padded_features ADD COLUMN CMEdec REAL")
cur.execute("ALTER TABLE padded_features ADD COLUMN CMEhis int")
cur.execute("ALTER TABLE padded_features ADD COLUMN CMEhis1d int")

for harpnum in tqdm(harpnums, desc="Calculating CME history parameters"):
    # Now we need need all the timestamps for the harpnum

    str_harpnum_timestamps = conn.execute(
        f"""
        SELECT padded_features.Timestamp
        FROM padded_features
        WHERE padded_features.harpnum = {harpnum}
        """
    ).fetchall()

    str_harpnum_timestamps = [x[0] for x in str_harpnum_timestamps]

    # And do the same thing
    harpnum_timestamps = np.array(
        [x for x in str_harpnum_timestamps], dtype="datetime64[s]"
    )[:, np.newaxis]

    # We'll do this by first selecting all the CMEs for the region
    flares = cmesrc.execute(
        f"""
        SELECT 
        C.cme_date
        FROM FINAL_CME_HARP_ASSOCIATIONS FCHA
        INNER JOIN CMES C ON C.cme_id = FCHA.cme_id
        WHERE harpnum = {harpnum}
        """
    ).fetchall()

    # Now put the timestamps into a numpy array as numpy datetime64

    timestamps = np.array([x[0] for x in flares], dtype="datetime64[s]")

    # Now calculate diffs which is timestamp - flare_time. In the end we should have a matrix of shape (n_harpnum_timestamps, n_flares)
    # the diffs must be in hours
    # And we use numpy for efficiency
    diffs = (harpnum_timestamps - timestamps) / np.timedelta64(1, "h")

    # Now we extract the N flares history parameters before replacing negative values with np.inf

    his_params = np.sum((diffs > 0).astype(int), axis=1).astype(int).tolist()
    his1d_params = (
        np.sum(((diffs < 24) & (diffs > 0)).astype(int), axis=1).astype(int).tolist()
    )

    # Need to replace negative values with np.inf
    diffs = np.where(diffs < 0, np.inf, diffs)

    # For the intensities and log intensities need to repeat them for each timestamp
    # So shape becomes (n_harpnum_timestamps, n_flares)

    fake_intensities = np.ones_like(diffs)
    fake_log_intensities = np.ones_like(diffs)

    (
        history_params,
        fake_intensity_params,
        fake_log_intensity_params,
    ) = calculate_history_params(diffs, 12, fake_intensities, fake_log_intensities)

    # Now update the database
    # Keep in mind history_params has shape (n_harpnum_timestamps) and we need to update the database for each timestamp
    cur.executemany(
        """
        UPDATE padded_features
        SET CMEdec = ?, CMEhis = ?, CMEhis1d = ?
        WHERE harpnum = ? AND Timestamp = ?
        """,
        zip(
            history_params,
            his_params,
            his1d_params,
            [harpnum] * len(history_params),
            str_harpnum_timestamps,
        ),
    )

# Can't normalize yet as I need to do train/test split first
conn.commit()
conn.close()
