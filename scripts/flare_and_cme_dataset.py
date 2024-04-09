import pulp
import sqlite3
from line_profiler import LineProfiler
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import pandas as pd
import sys
import numpy as np
import numpy.typing as npt
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
from tqdm import tqdm
from typing import List, Tuple, Union, NamedTuple, Optional, Any

CMESRC_V3 = "/home/julio/cmesrc/data/processed/final_results/cmesrcV3.db"
FEATURES = "/home/julio/cmeml_experiments/experiment2/data/features.db"

output_figs_dir = "/home/julio/cmeml_experiments/experiment2/logs/dataset_figs/"


def setup_database(cur, final_table_name):
    cur.execute(f"DROP TABLE IF EXISTS {final_table_name}")
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {final_table_name} (
        harpnum INT, 
        period_start TEXT,
        period_end TEXT,
        n_points INT,
        activity_in_obs INT,
        has_flare_above_threshold INT,
        has_flare_below_threshold INT,
        flare_above_threshold_id INT,
        flare_below_threshold_id INT,
        has_cme_flare_above_threshold INT,
        has_cme_flare_below_threshold INT,
        has_cme_no_flare INT,
        cme_flare_above_threshold_id INT,
        cme_flare_below_threshold_id INT,
        cme_no_flare_id INT,
        split INT
        )"""
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS DATASETS (
            dataset_name TEXT PRIMARY KEY,
            last_updated TEXT,
            T REAL,
            L REAL,
            S REAL,
            B REAL,
            ALLOW_OVERLAPS INT,
            MIN_FLARE_CLASS INT,
            N_TOTAL INT,
            N_DROPPED_NOT_ENOUGH_POINTS INT,
            N_DROPPER_OVERLAP INT
        )
        """
    )

    # Create useful indices
    cur.execute("CREATE INDEX IF NOT EXISTS harpnum_index ON padded_features(harpnum)")


def create_indices():
    conn = sqlite3.connect(CMESRC_V3)
    cur = conn.cursor()

    cur.execute(
        "CREATE INDEX IF NOT EXISTS FCHA_cme_id_index ON FINAL_CME_HARP_ASSOCIATIONS(cme_id)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS CME_cme_id_index ON CMES(cme_id)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS CME_HARPS_EVENTS_cme_id_index ON CMES_HARPS_EVENTS(cme_id)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS FLARES_flare_id_index ON FLARES(flare_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS FLARES_harpnum_index ON FLARES(harpnum)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS FLARES_flare_date_index ON FLARES(flare_date)"
    )


# Function to parse command line arguments into a dictionary
def parse_arguments():
    args = sys.argv[1:]
    args_dict = {}
    for arg in args:
        key_value = arg.split("=")
        if len(key_value) == 2:
            args_dict[key_value[0]] = key_value[1]
    return args_dict


class HarpsDatasetSlices:
    def __init__(
        self,
        harpnum: int,
        T: float,
        L: float,
        S: float,
        B: float,
        ALLOW_OVERLAPS: bool,
        MIN_FLARE_CLASS: int,
        cur: sqlite3.Cursor,
    ):
        # Initialize basic attributes
        self.harpnum = harpnum
        self.T = T
        self.S = S
        self.L = L
        self.B = B
        self.cur = cur

        self.ALLOW_OVERLAPS = ALLOW_OVERLAPS

        self.N_DROP_NOT_ENOUGH_POINTS = 0
        self.N_DROP_OVERLAP = 0

        self.expected_points = self.L * 5

        cur.execute(
            "SELECT MIN(Timestamp) FROM padded_features WHERE harpnum = ? LIMIT 1",
            (self.harpnum,),
        )
        self.first_date = datetime.strptime(cur.fetchone()[0], "%Y-%m-%d %H:%M:%S")

        # To avoid issues, substract five minutes from first date

        self.first_date = self.first_date - timedelta(minutes=5)

        cur.execute(
            "SELECT MAX(Timestamp) FROM padded_features WHERE harpnum = ? LIMIT 1",
            (self.harpnum,),
        )
        self.last_date = datetime.strptime(cur.fetchone()[0], "%Y-%m-%d %H:%M:%S")

        self.flare_data = cur.execute(
            """
        WITH CME_FLARES AS (
        SELECT
        FCHA.cme_id,
        FCHA.harpnum,
        CHE.flare_id
        FROM cmesrc.FINAL_CME_HARP_ASSOCIATIONS FCHA
        INNER JOIN cmesrc.CMES_HARPS_EVENTS CHE ON FCHA.cme_id = CHE.cme_id AND FCHA.harpnum = CHE.harpnum
        )

        SELECT 
        F.flare_id,
        F.flare_date, 
        F.flare_class, 
        F.flare_class_score,
        CF.cme_id
        FROM cmesrc.flares F
        LEFT JOIN CME_FLARES CF ON F.flare_id = CF.flare_id
        WHERE F.harpnum = ? ORDER BY F.flare_date ASC
        """,
            (self.harpnum,),
        )

        self.MIN_FLARE_CLASS = MIN_FLARE_CLASS

        self.flare_data = self.flare_data.fetchall()

        flare_dates_strings = [data[1] for data in self.flare_data]

        self.flare_dates = np.array(flare_dates_strings, dtype="datetime64").astype(
            datetime
        )
        self.flare_ids = np.array([data[0] for data in self.flare_data], dtype=int)
        self.flare_classes = np.array([data[2] for data in self.flare_data])
        self.flare_class_scores = np.array([data[3] for data in self.flare_data])
        self.flare_cme_ids = np.array(
            [np.nan if i[4] is None else i[4] for i in self.flare_data]
        )

        self.cme_data = cur.execute(
            """
        WITH CME_FLARES AS (
        SELECT
        FCHA.cme_id,
        C.cme_date,
        FCHA.harpnum,
        CHE.flare_id
        FROM cmesrc.FINAL_CME_HARP_ASSOCIATIONS FCHA
        INNER JOIN cmesrc.CMES_HARPS_EVENTS CHE ON FCHA.cme_id = CHE.cme_id AND FCHA.harpnum = CHE.harpnum
        INNER JOIN cmesrc.CMES C ON FCHA.cme_id = C.cme_id
        )

        SELECT
        cme_id,
        cme_date
        FROM CME_FLARES WHERE harpnum = ? AND flare_id IS NULL ORDER BY cme_date ASC
        """,
            (self.harpnum,),
        )

        # Convert this list of tuples to a numpy array
        self.cme_data = self.cme_data.fetchall()

        cme_dates_strings = [data[1] for data in self.cme_data]

        self.cme_dates = np.array(cme_dates_strings, dtype="datetime64").astype(
            datetime
        )
        self.cme_ids = np.array([data[0] for data in self.cme_data], dtype=int)

    def check_if_event_within_period(
        self,
        period: Tuple[datetime, datetime],
    ) -> bool:
        period_start, period_end = period

        # Need to adjust the period start by the buffer B
        adjusted_period_start = period_start - timedelta(hours=self.B)

        # Find the insertion points for period_start and period_end
        flares_start_index = bisect_left(self.flare_dates, adjusted_period_start)
        flares_end_index = bisect_right(self.flare_dates, period_end)

        cmes_start_index = bisect_left(self.cme_dates, adjusted_period_start)
        cmes_end_index = bisect_right(self.cme_dates, period_end)

        # Check if there is an overlap
        return (flares_start_index < flares_end_index) or (
            cmes_start_index < cmes_end_index
        )

    def get_labels(
        self, ref_time: datetime
    ) -> Tuple[
        bool,
        bool,
        Union[int, None],
        Union[int, None],
        bool,
        bool,
        bool,
        Union[int, None],
        Union[int, None],
        Union[int, None],
    ]:
        # Define the start and end of the time period for analysis
        per_start, per_end = ref_time, ref_time + timedelta(hours=self.T)

        # Find indices in the flare_dates array that correspond to the defined time window
        start_index = bisect_left(self.flare_dates, per_start)
        end_index = bisect_right(self.flare_dates, per_end)

        if end_index == len(self.flare_dates):
            end_index -= 1

        # Initialize variables to track the presence of flares and their IDs
        has_flare_above_threshold = has_flare_below_threshold = False
        flare_above_threshold_id = flare_below_threshold_id = None
        has_flare_below_threshold = has_flare_below_threshold = False
        flare_below_threshold_id = flare_below_threshold_id = None

        # Initialize variables to track the presence of CMEs and their associated flare and CME IDs
        has_cme_flare_above_threshold = (
            has_cme_flare_below_threshold
        ) = has_cme_no_flare = False
        cme_flare_above_threshold_id = (
            cme_flare_below_threshold_id
        ) = cme_no_flare_id = None

        # Check if there are any flares within the defined time window
        if start_index < end_index:
            # Extract the class scores of flares within the time window
            valid_flare_class_scores = self.flare_class_scores[start_index:end_index]
            valid_flare_ids = self.flare_ids[start_index:end_index]
            valid_flare_cme_ids = self.flare_cme_ids[start_index:end_index]

            threshold_mask = valid_flare_class_scores >= self.MIN_FLARE_CLASS
            cmes_mask = ~np.isnan(valid_flare_cme_ids)

            # Now a trick, if cme_mask is all False, then we want to set it to all True
            if np.all(~cmes_mask):
                cmes_mask = ~cmes_mask

            above_mask = threshold_mask & cmes_mask
            below_mask = (~threshold_mask) & cmes_mask

            # If there's a CME only associated with a flare either above or below
            # The above will set the mask to all False, meaning even if there's a flare
            # it's ignored. Check if all is false, just set the mask to all true

            if np.all(~above_mask):
                above_mask = ~above_mask
                above_mask = above_mask & threshold_mask

            if np.all(~below_mask):
                below_mask = ~below_mask
                below_mask = below_mask & (~threshold_mask)

            # Check for flares above the class threshold
            if np.any(above_mask):
                has_flare_above_threshold = True

                flare_above_threshold_id = valid_flare_ids[above_mask][0]
                cme_flare_above_threshold_id = valid_flare_cme_ids[above_mask][0]

                # Now because of our trick above, cme_id could be nan, so we need to check for that
                if np.isnan(cme_flare_above_threshold_id):
                    cme_flare_above_threshold_id = None
                    has_cme_flare_above_threshold = False
                else:
                    cme_flare_above_threshold_id = int(cme_flare_above_threshold_id)
                    has_cme_flare_above_threshold = True

            # Check for flares below the class threshold
            if np.any(below_mask):
                has_flare_below_threshold = True

                flare_below_threshold_id = valid_flare_ids[below_mask][0]
                cme_flare_below_threshold_id = valid_flare_cme_ids[below_mask][0]

                if np.isnan(cme_flare_below_threshold_id):
                    cme_flare_below_threshold_id = None
                    has_cme_flare_below_threshold = False
                else:
                    cme_flare_below_threshold_id = int(cme_flare_below_threshold_id)
                    has_cme_flare_below_threshold = True

        # Now need to repeat for CMEs without flares
        # Find indices in the cme_dates array that correspond to the defined time window
        cme_nf_start_index = bisect_left(self.cme_dates, per_start)
        cme_nf_end_index = bisect_right(self.cme_dates, per_end)

        if cme_nf_end_index == len(self.cme_dates):
            cme_nf_end_index -= 1

        # Check if there are any flares within the defined time window
        if cme_nf_start_index < cme_nf_end_index:
            # Then there's a CME without flares
            has_cme_no_flare = True
            cme_no_flare_id = self.cme_ids[cme_nf_start_index]

        return (
            has_flare_above_threshold,
            has_flare_below_threshold,
            flare_above_threshold_id,
            flare_below_threshold_id,
            has_cme_flare_above_threshold,
            has_cme_flare_below_threshold,
            has_cme_no_flare,
            cme_flare_above_threshold_id,
            cme_flare_below_threshold_id,
            cme_no_flare_id,
        )

    def process(self):
        def step(period_start, period_end) -> Tuple[datetime, datetime]:
            return (
                period_start + timedelta(hours=self.S),
                period_end + timedelta(hours=self.S),
            )

        def should_keep_going(period_end) -> bool:
            return period_end < self.last_date

        single_result_type = Tuple[
            int,  # harpnum
            str,  # period_start
            str,  # period_end
            int,  # n_valid_points
            int,  # activity_in_obs
            int,  # has_flare_above_threshold
            int,  # has_flare_below_threshold
            Union[int, None],  # flare_above_threshold_id
            Union[int, None],  # flare_below_threshold_id
            int,  # has_cme_flare_above_threshold
            int,  # has_cme_flare_below_threshold
            int,  # has_cme_no_flare
            Union[int, None],  # cme_flare_above_threshold_id
            Union[int, None],  # cme_flare_below_threshold_id
            Union[int, None],  # cme_no_flare_id
        ]
        results_type = List[single_result_type]

        results: results_type = []
        # flare_id, period_start, period_end, n_valid_points, cme_id, label

        period_start = self.first_date
        period_end = self.first_date + timedelta(hours=self.L)

        keep_going = True

        while keep_going:
            # First let's check if there's a flare or CME of any kind within the period
            activity_in_obs = self.check_if_event_within_period(
                (period_start, period_end)
            )

            if activity_in_obs and not self.ALLOW_OVERLAPS:
                period_start, period_end = step(period_start, period_end)
                self.N_DROP_OVERLAP += 1
                keep_going = should_keep_going(period_end)
                continue

            # If there's no flare, we need to check if there's one in the T hours
            # after the period_end

            # Get the labels

            (
                has_flare_above_threshold,
                has_flare_below_threshold,
                flare_above_threshold_id,
                flare_below_threshold_id,
                has_cme_flare_above_threshold,
                has_cme_flare_below_threshold,
                has_cme_no_flare,
                cme_flare_above_threshold_id,
                cme_flare_below_threshold_id,
                cme_no_flare_id,
            ) = self.get_labels(period_end)

            # Get the number of valid points
            NPOINTS: int = self.cur.execute(
                """
            SELECT COUNT(*) FROM padded_features
            WHERE
            harpnum = ? AND
            Timestamp between ? AND ? AND
            IS_VALID = 1
            """,
                (
                    int(self.harpnum),
                    period_start.strftime("%Y-%m-%d %H:%M:%S"),
                    period_end.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            ).fetchone()[0]

            # Check last timestamp is valid

            LAST_TIMESTAMP_VALID: bool = bool(
                self.cur.execute(
                    """
            SELECT IS_VALID FROM padded_features
            WHERE
            harpnum = ? AND
            Timestamp between ? AND ?
            ORDER BY Timestamp DESC
            LIMIT 1
            """,
                    (
                        int(self.harpnum),
                        period_start.strftime("%Y-%m-%d %H:%M:%S"),
                        period_end.strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                ).fetchone()[0]
            )

            TOTALPOINTS: int = self.cur.execute(
                """
            SELECT COUNT(*) FROM padded_features
            WHERE
            harpnum = ? AND
            Timestamp between ? AND ?
            """,
                (
                    int(self.harpnum),
                    period_start.strftime("%Y-%m-%d %H:%M:%S"),
                    period_end.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            ).fetchone()[0]

            # Last timestamp has to be valid (to use as query, etc...)

            if not LAST_TIMESTAMP_VALID:
                self.N_DROP_NOT_ENOUGH_POINTS += 1
                period_start, period_end = step(period_start, period_end)
                keep_going = should_keep_going(period_end)
                continue

            # Total points must equal the expected_points
            # because otherwise the region was just non
            # existent for part of the period

            if TOTALPOINTS != self.expected_points:
                self.N_DROP_NOT_ENOUGH_POINTS += 1
                period_start, period_end = step(period_start, period_end)
                keep_going = should_keep_going(period_end)
                continue

            PERCENT_POINTS: float = NPOINTS / self.expected_points

            if PERCENT_POINTS < 0.8:
                self.N_DROP_NOT_ENOUGH_POINTS += 1
                period_start, period_end = step(period_start, period_end)
                keep_going = should_keep_going(period_end)
                continue

            new_result: single_result_type = (
                self.harpnum,
                period_start.strftime("%Y-%m-%d %H:%M:%S"),
                period_end.strftime("%Y-%m-%d %H:%M:%S"),
                int(NPOINTS),
                int(activity_in_obs),
                int(has_flare_above_threshold),
                int(has_flare_below_threshold),
                int(flare_above_threshold_id) if flare_above_threshold_id else None,
                int(flare_below_threshold_id) if flare_below_threshold_id else None,
                int(has_cme_flare_above_threshold),
                int(has_cme_flare_below_threshold),
                int(has_cme_no_flare),
                int(cme_flare_above_threshold_id)
                if cme_flare_above_threshold_id
                else None,
                int(cme_flare_below_threshold_id)
                if cme_flare_below_threshold_id
                else None,
                int(cme_no_flare_id) if cme_no_flare_id else None,
            )

            results.append(new_result)
            period_start, period_end = step(period_start, period_end)
            keep_going = should_keep_going(period_end)

        return results


def update_final_table(cur, final_table_name, harpnum, results):
    for result in results:
        cur.execute(
            f"""
        INSERT INTO {final_table_name} (
        harpnum, 
        period_start,
        period_end,
        n_points,
        activity_in_obs,
        has_flare_above_threshold,
        has_flare_below_threshold,
        flare_above_threshold_id,
        flare_below_threshold_id,
        has_cme_flare_above_threshold,
        has_cme_flare_below_threshold,
        has_cme_no_flare,
        cme_flare_above_threshold_id,
        cme_flare_below_threshold_id,
        cme_no_flare_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (*result,),
        )


def update_dataset_summary(
    cur: sqlite3.Cursor,
    FINAL_TABLE_NAME: str,
    T: float,
    L: float,
    S: float,
    B: float,
    ALLOW_OVERLAPS: bool,
    MIN_FLARE_CLASS: int,
    N_DROP_NOT_ENOUGH_POINTS: int,
    N_DROP_OVERLAP: int,
):
    # The schema is:
    #     CREATE TABLE IF NOT EXISTS DATASETS (
    #         dataset_name TEXT PRIMARY KEY,
    #         last_updated TEXT,
    #         T REAL,
    #         L REAL,
    #         S REAL,
    #         B REAL,
    #         MIN_FLARE_CLASS INT,
    #         N_TOTAL INT,
    #         N_DROPPED_NOT_ENOUGH_POINTS INT<
    #         N_DROPPER_OVERLAP INT
    #     )

    # Get the current time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get the number of positive and negative examples

    N_TOTAL = cur.execute(
        f"""
        SELECT COUNT(*) FROM {FINAL_TABLE_NAME}
        """
    ).fetchone()[0]

    # Insert the new row

    cur.execute(
        """
    INSERT OR REPLACE INTO DATASETS (
    dataset_name,
    last_updated,
    T,
    L,
    S,
    B,
    ALLOW_OVERLAPS,
    MIN_FLARE_CLASS,
    N_TOTAL,
    N_DROPPED_NOT_ENOUGH_POINTS,
    N_DROPPER_OVERLAP
        )

    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            FINAL_TABLE_NAME,
            now,
            T,
            L,
            S,
            B,
            int(ALLOW_OVERLAPS),
            MIN_FLARE_CLASS,
            N_TOTAL,
            N_DROP_NOT_ENOUGH_POINTS,
            N_DROP_OVERLAP,
        ),
    )


def format_float_for_table_name(f):
    if f.is_integer():
        return str(int(f))
    else:
        return str(f).replace(".", "_")


def create_splits(conn, final_table_name, N_SPLITS, time_limit=600):
    # SQL query to fetch event counts and total rows
    query = f"""
    SELECT
        harpnum,
        COUNT(DISTINCT(flare_above_threshold_id)) AS flares_1,
        COUNT(DISTINCT(cme_flare_above_threshold_id)) AS cmes_flares_1,
        COUNT(DISTINCT(
            CASE
                WHEN cme_flare_above_threshold_id IS NOT NULL THEN cme_flare_above_threshold_id
                WHEN cme_flare_below_threshold_id IS NOT NULL THEN cme_flare_below_threshold_id
                ELSE cme_no_flare_id
            END
        )) AS cmes_total_1,
        COUNT(*) AS n_rows
    FROM
        {final_table_name}
    GROUP BY
        harpnum;
    """

    # Fetch data
    counts = pd.read_sql(query, conn)

    # Set up the ILP problem
    problem = pulp.LpProblem("SplitOptimization", pulp.LpMinimize)

    # Decision variables: whether a harpnum is in a particular split
    split_vars = pulp.LpVariable.dicts(
        "Split",
        ((i, harpnum) for i in range(N_SPLITS) for harpnum in counts["harpnum"]),
        cat="Binary",
    )

    # Auxiliary variables for absolute differences
    abs_diff_vars = pulp.LpVariable.dicts(
        "AbsDiff",
        (
            (i, event_type)
            for i in range(N_SPLITS)
            for event_type in ["flares_1", "cmes_flares_1", "cmes_total_1"]
        ),
        lowBound=0,
    )

    # Objective: Minimize the sum of absolute differences from the mean for event counts across splits
    problem += pulp.lpSum(
        [
            abs_diff_vars[i, event_type]
            for i in range(N_SPLITS)
            for event_type in ["flares_1", "cmes_flares_1", "cmes_total_1"]
        ]
    )

    for event_type in ["flares_1", "cmes_flares_1", "cmes_total_1"]:
        event_counts = [
            pulp.lpSum(
                counts[counts["harpnum"] == harpnum][event_type].iloc[0]
                * split_vars[i, harpnum]
                for harpnum in counts["harpnum"]
            )
            for i in range(N_SPLITS)
        ]
        average_event_count = pulp.lpSum(event_counts) / N_SPLITS

        # Constraints for absolute differences
        for i in range(N_SPLITS):
            problem += (
                abs_diff_vars[i, event_type] >= event_counts[i] - average_event_count
            )
            problem += abs_diff_vars[i, event_type] >= -(
                event_counts[i] - average_event_count
            )

    # Each harpnum must be in exactly one split
    for harpnum in counts["harpnum"]:
        problem += (
            pulp.lpSum(split_vars[i, harpnum] for i in range(N_SPLITS)) == 1,
            f"OneSplit_{harpnum}",
        )

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)  # Set the time limit here

    # Solve the problem
    problem.solve(solver)

    # Process the results
    split_results = [[] for _ in range(N_SPLITS)]
    harpnum_assignment_count = (
        {}
    )  # To track the number of splits each harpnum is assigned to
    for i in range(N_SPLITS):
        for harpnum in counts["harpnum"]:
            if pulp.value(split_vars[i, harpnum]) == 1:
                split_results[i].append(harpnum)
                harpnum_assignment_count[harpnum] = (
                    harpnum_assignment_count.get(harpnum, 0) + 1
                )

    for harpnum, count in harpnum_assignment_count.items():
        if count != 1:
            raise ValueError(
                f"Harpnum {harpnum} is assigned to {count} splits. It should only be in one split."
            )

    # Gather statistics
    split_counts = np.zeros((N_SPLITS, 4))
    for i, harpnums in enumerate(split_results):
        for harpnum in harpnums:
            row = counts[counts["harpnum"] == harpnum].iloc[0]
            split_counts[i] += row[1:]

    split_lens = np.array([len(split) for split in split_results])
    print("Split counts:")
    for split_idx, split_count in enumerate(split_counts):
        print(f"Split {split_idx}: {split_count}, N_harpnums: {split_lens[split_idx]}")
    print("Statistics:")
    print(
        f"Splits: mean: {split_counts.mean(axis=0)}, std: {split_counts.std(axis=0)}. N Harpnums mean: {split_lens.mean()}, std: {split_lens.std()}"
    )

    return split_results, split_counts


def update_with_splits(conn, final_table_name, N_SPLITS):
    split_harpnums, split_counts = create_splits(conn, final_table_name, N_SPLITS)

    conn.execute(f"DROP TABLE IF EXISTS {final_table_name}_splits")
    conn.execute(
        f"CREATE TABLE {final_table_name}_splits (split INT, is_active INT, is_cme_active, harpnum INT PRIMARY KEY)"
    )
    conn.execute(
        f"CREATE INDEX {final_table_name}_harpnum_index ON {final_table_name} (harpnum)"
    )

    for split_idx, harpnums in enumerate(split_harpnums):
        values = [(split_idx, harpnum) for harpnum in harpnums]
        conn.executemany(
            f"UPDATE {final_table_name} SET split = ? WHERE harpnum = ?", values
        )
        conn.executemany(
            f"INSERT INTO {final_table_name}_splits (split, harpnum) VALUES (?, ?)",
            values,
        )

    # This is now to add the is_active column
    query = f"""
    CREATE TEMPORARY TABLE TempIsActive AS
    SELECT 
        DTS.harpnum,
        CASE 
            WHEN MAX(MDS.has_flare_above_threshold) = 1 OR MAX(MDS.has_flare_below_threshold) = 1 THEN 1
            ELSE 0
        END AS isActive,
        CASE
            WHEN MAX(MDS.has_cme_flare_above_threshold) = 1 OR MAX(MDS.has_cme_flare_below_threshold) = 1 OR MAX(MDS.has_cme_no_flare) = 1 THEN 1
            ELSE 0
        END AS isCMEActive
    FROM 
        {final_table_name}_splits AS DTS
    INNER JOIN 
        {final_table_name} AS MDS ON DTS.harpnum = MDS.harpnum
    GROUP BY 
        DTS.harpnum;

    UPDATE {final_table_name}_splits
    SET 
        is_active = (
            SELECT isActive
            FROM TempIsActive
            WHERE {final_table_name}_splits.harpnum = TempIsActive.harpnum
        ), 
        is_cme_active = (
            SELECT isCMEActive
            FROM TempIsActive
            WHERE {final_table_name}_splits.harpnum = TempIsActive.harpnum
    );

    DROP TABLE TempIsActive;
    """

    conn.executescript(query)

    plot_split_statistics(conn, final_table_name, N_SPLITS)


def plot_split_statistics(conn, final_table_name, N_SPLITS):
    # Initialize lists for all data points
    n_harpnums_in_act_per_split = []
    n_harpnums_not_in_act_per_split = []
    n_flares_above_per_split = []
    n_flares_below_per_split = []
    n_cmes_flare_above_per_split = []
    n_cmes_flare_below_per_split = []
    n_cmes_no_flare_per_split = []
    n_rows_per_split = []

    for split_idx in range(N_SPLITS):
        query_result = conn.execute(
            f"""
            WITH regions_with_act AS (
                SELECT harpnum, split
                FROM {final_table_name}
                WHERE has_flare_above_threshold = 1 OR 
                    has_flare_below_threshold = 1 OR 
                    has_cme_flare_above_threshold = 1 OR 
                    has_cme_flare_below_threshold = 1 OR 
                    has_cme_no_flare = 1
            ),
            aggregated_act AS (
                SELECT split, COUNT(DISTINCT harpnum) AS count_harpnums_in_act
                FROM regions_with_act
                GROUP BY split
            )
            SELECT 
                COUNT(DISTINCT T.flare_above_threshold_id) AS n_flares_above,
                COUNT(DISTINCT T.flare_below_threshold_id) AS n_flares_below,
                COUNT(DISTINCT T.cme_flare_above_threshold_id) AS n_cmes_flare_above,
                COUNT(DISTINCT T.cme_flare_below_threshold_id) AS n_cmes_flare_below,
                COUNT(DISTINCT T.cme_no_flare_id) AS n_cmes_no_flare,
                COUNT(DISTINCT T.harpnum) AS n_harpnums,
                COUNT(*) AS n_rows,
                COALESCE(A.count_harpnums_in_act, 0) AS n_harpnums_in_act,
                (COUNT(DISTINCT T.harpnum) - COALESCE(A.count_harpnums_in_act, 0)) AS n_harpnums_not_in_act
            FROM 
                {final_table_name} T
            LEFT JOIN 
                aggregated_act A ON T.split = A.split
            WHERE T.split = ?
            GROUP BY 
                T.split
                """,
            (split_idx,),
        ).fetchall()[0]

        # Append the new data to respective lists
        n_flares_above_per_split.append(query_result[0])
        n_flares_below_per_split.append(query_result[1])
        n_cmes_flare_above_per_split.append(query_result[2])
        n_cmes_flare_below_per_split.append(query_result[3])
        n_cmes_no_flare_per_split.append(query_result[4])
        n_rows_per_split.append(query_result[6])
        n_harpnums_in_act_per_split.append(query_result[7])
        n_harpnums_not_in_act_per_split.append(query_result[8])

    # Define figure and subplot layout
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 30))
    fig.suptitle(f"Statistics for Each Split in {final_table_name}", fontsize=16)

    # Helper function for plotting
    def plot_bar(ax, data, title, color, xlabel="Split Index", ylabel="Count"):
        ax.bar(np.arange(N_SPLITS), data, color=color)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Plotting each data point in a subplot
    plot_bar(axes[0, 0], n_flares_above_per_split, "Flares Above Threshold", "orange")
    plot_bar(axes[0, 1], n_flares_below_per_split, "Flares Below Threshold", "yellow")
    plot_bar(
        axes[1, 0],
        n_cmes_flare_above_per_split,
        "CMEs with Flare Above Threshold",
        "red",
    )
    plot_bar(
        axes[1, 1],
        n_cmes_flare_below_per_split,
        "CMEs with Flare Below Threshold",
        "pink",
    )
    plot_bar(axes[2, 0], n_cmes_no_flare_per_split, "CMEs without Flare", "purple")
    plot_bar(axes[2, 1], n_rows_per_split, "Total Rows per Split", "blue")
    plot_bar(
        axes[3, 0], n_harpnums_in_act_per_split, "Active Harpnums per Split", "green"
    )
    plot_bar(
        axes[3, 1],
        n_harpnums_not_in_act_per_split,
        "Non-Active Harpnums per Split",
        "grey",
    )

    # Improving the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_figs_dir}/{final_table_name}_split_statistics.png")


# Main function
def main(TESTING: bool):
    args_dict = parse_arguments()

    T = float(args_dict["T"])
    L = float(args_dict["L"])
    S = float(args_dict["S"])
    B = float(args_dict["B"])
    MIN_FLARE_CLASS = int(args_dict["MIN_FLARE_CLASS"])
    ALLOW_OVERLAPS = bool(int(args_dict["ALLOW_OVERLAPS"]))

    formatted_T = format_float_for_table_name(T)
    formatted_L = format_float_for_table_name(L)
    formatted_S = format_float_for_table_name(S)
    formatted_B = format_float_for_table_name(B)

    create_indices()

    conn = sqlite3.connect(FEATURES)
    cur = conn.cursor()
    cur.execute(f"ATTACH DATABASE '{CMESRC_V3}' AS cmesrc")

    FINAL_TABLE_NAME = f"""T_{formatted_T}_L_{formatted_L}_S_{formatted_S}_B_{formatted_B}_OVERLAPS_{int(ALLOW_OVERLAPS)}_MIN_FLARE_{MIN_FLARE_CLASS}_dataset"""

    if TESTING:
        FINAL_TABLE_NAME = "TESTING_TABLE"

    # Database setup and logging configuration
    setup_database(cur, FINAL_TABLE_NAME)
    harpnums = cur.execute("SELECT DISTINCT harpnum FROM padded_features").fetchall()
    harpnums = [harpnum[0] for harpnum in harpnums]

    TOTAL_N_DROP_NOT_ENOUGH_POINTS = 0
    TOTAL_N_DROP_OVERLAP = 0

    if TESTING:
        # harpnums = harpnums[:300]
        harpnums = [3291]

    for harpnum in tqdm(harpnums):
        harp_dataset = HarpsDatasetSlices(
            harpnum, T, L, S, B, ALLOW_OVERLAPS, MIN_FLARE_CLASS, cur
        )

        results = harp_dataset.process()

        TOTAL_N_DROP_NOT_ENOUGH_POINTS += harp_dataset.N_DROP_NOT_ENOUGH_POINTS
        TOTAL_N_DROP_OVERLAP += harp_dataset.N_DROP_OVERLAP

        update_final_table(cur, FINAL_TABLE_NAME, harpnum, results)

    if not TESTING:
        update_with_splits(conn, FINAL_TABLE_NAME, 10)

    if not TESTING:
        # Update dataset summary
        update_dataset_summary(
            cur,
            FINAL_TABLE_NAME,
            T,
            L,
            S,
            B,
            ALLOW_OVERLAPS,
            MIN_FLARE_CLASS,
            TOTAL_N_DROP_NOT_ENOUGH_POINTS,
            TOTAL_N_DROP_OVERLAP,
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    TESTING = False
    if TESTING:
        profiler = LineProfiler()
        # profiler.add_function(main)
        # profiler.add_function(HarpsDatasetSlices.__init__)

        profiler.runcall(main, TESTING=TESTING)

        profiler.dump_stats("dataset_profiler.lprof")
        profiler.print_stats()
    else:
        main(TESTING=False)
