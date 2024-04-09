from .base_dataset import BaseDataset
from cuml import PCA as cuPCA
import pandas as pd
import numpy as np
import torch
import threading
from ..utils.global_signals import cleanup_event
from sklearn.metrics import confusion_matrix, average_precision_score
import os
import sqlite3
from typing import Union, Optional
import uuid

FLARE_CLASSES = ["B", "C", "M", "X"]

FLARE_HISTORY_KEYWORDS = []

for flare_class in FLARE_CLASSES:
    FLARE_HISTORY_KEYWORDS.append(f"{flare_class}dec")
    FLARE_HISTORY_KEYWORDS.append(f"{flare_class}his")
    FLARE_HISTORY_KEYWORDS.append(f"{flare_class}his1d")

INTENSITY_CLASSES = ["E", "logE"]
INTENSITY_KEYWORDS = []

for intensity_class in INTENSITY_CLASSES:
    INTENSITY_KEYWORDS.append(f"{intensity_class}dec")


CME_HISTORY_KEYWORDS = ["CMEdec", "CMEhis", "CMEhis1d"]

DEFAULT_KEYWORDS = [  # Lists all the available keywords
    "ABSNJZH",
    "EPSX",
    "EPSY",
    "EPSZ",
    "MEANALP",
    "MEANGAM",
    "MEANGBH",
    "MEANGBT",
    "MEANGBZ",
    "MEANJZD",
    "MEANJZH",
    "MEANPOT",
    "MEANSHR",
    "R_VALUE",
    "SAVNCPP",
    "SHRGT45",
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "TOTPOT",
    "TOTUSJH",
    "TOTUSJZ",
    "USFLUX",
]

DEFAULT_KEYWORDS += FLARE_HISTORY_KEYWORDS
DEFAULT_KEYWORDS += INTENSITY_KEYWORDS
DEFAULT_KEYWORDS += CME_HISTORY_KEYWORDS


def format_float_for_table_name(f):
    if not isinstance(f, float):
        f = float(f)
    if f.is_integer():
        return str(int(f))
    else:
        return str(f).replace(".", "_")


class PreProcessSWANDataset:
    """
    This class does pre-processing of the SWAN dataset.

    In particular, it does the following:

    - Creates a "temporary" copy of the features table
      selecting only the relevant splits. Stores it as
      a normal table with an unique name.

    - Applies padding to the temp. features table.

    - Normalizes the features to mean 0 and std 1
      This is either using given means and stds or
      using the rows directly.

    - Returns the name of the temp. features table
    """

    def __init__(
        self,
        mode: str,  # Mode of the dataset. Either train or validation (which includes testing)
        db_path: str,  # Path to features db
        temp_db_id: str,  # Unique id of the temporary database
        splits: list,  # List of splits to use in the databse (for cross-validation and splitting into training/test)
        T: float,  # Flare happens within next T hours. Does it have CME?
        L: float,  # We take observations of length L hours so end of observation within T to flare
        S: float,  # Observations are S hours apart
        B: float,  # We don't allow flares within observations. We also consider B hours of buffer after the flare as non ok.
        C: float,  # This is the cadence, so data points are taken over a period of L hours with each separated by C hours
        MIN_FLARE_CLASS: float,  # Only flares of class larger than this. For example, 35 is M5
        ALLOW_OVERLAPS: bool,  # If this is false, we allow flares in the observation period
        use_only_active: bool = False,  # If true, only use regions that every produced a flare
        padding_strategy: str = "zero",
        keywords: list = DEFAULT_KEYWORDS,  # List of keywords to use. DEFAULT_KEYWORDS is a list of all available keywords
        not_keywords: list = None,  # List of keywords to not use. If None, all keywords are used.
        features_means: Union[
            dict, None
        ] = None,  # Means of the keywords. To use when getting validation or testing sets.
        features_stds: Union[dict, None] = None,  # Same as means
        use_PCA: bool = False,
        n_PCA_components: Optional[int] = None,
        PCA_object: Optional[cuPCA] = None,
        use_is_valid: bool = False,
        *args,  # Any other
        **kwargs,  # Any other
    ):
        try:
            self._initialize_parameters(
                mode,
                db_path,
                splits,
                T,
                L,
                S,
                B,
                C,
                MIN_FLARE_CLASS,
                ALLOW_OVERLAPS,
                keywords,
                not_keywords,
                padding_strategy,
                features_means,
                features_stds,
                use_only_active,
                temp_db_id,
                use_PCA,
                n_PCA_components,
                PCA_object,
                use_is_valid,
            )

            self._setup_dataset()

            # self.cleanup_thread = threading.Thread(target=self.wait_for_cleanup)
            # self.cleanup_thread.start()
        except Exception as e:
            self.cleanup()
            raise e

    def get_output(self):
        """
        Returns the name of the temporary table and the means and stds
        """
        return (
            self.temp_table_name,
            (
                self.calculated_features_means,
                self.calculated_features_stds,
            ),
            self.temp_db_path,
            self.PCA,
            self.pca_columns,
        )

    def _initialize_parameters(
        self,
        mode: str,
        db_path: str,
        splits: list,
        T: float,
        L: float,
        S: float,
        B: float,
        C: float,
        MIN_FLARE_CLASS: float,
        ALLOW_OVERLAPS: bool,
        keywords: list,
        not_keywords: list,
        padding_strategy: str,
        features_means: Union[dict, None],
        features_stds: Union[dict, None],
        use_only_active: bool,
        temp_db_id: str,
        use_PCA: bool,
        n_PCA_components: Optional[int],
        PCA_object: Optional[cuPCA],
        use_is_valid,
    ):
        """
        Initialize the parameters of the dataset.
        """
        # Mode should either be train or val
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Mode {mode} is not valid. Must be either train, val or test"

        # Check T, L, S and B are positive
        assert T > 0, "T must be > 0"
        assert L > 0, "L must be > 0"
        assert S > 0, "S must be > 0"
        assert B >= 0, "B must be >= 0"
        assert MIN_FLARE_CLASS > 0, "MIN_FLARE_CLASS must be positive"

        # Cadence is a bit more involved, as it must be a multiple of 12 minutes
        # Therefore, in hours it must be a multiple of 0.2

        assert C > 0, "C must be > 0"
        assert (
            C % 0.2 == 0
        ), "C must be a multiple of 0.2. This is because the data is taken every 12 minutes"

        # And now IMPORTANT. We store C in indices, not in hours
        # so we can then use it to index the data
        self.C = int(C / 0.2)

        # With this we can calculate the expected shape of the data

        self.output_shape = (int(L / C), len(keywords))

        # Check db path exists
        assert os.path.exists(db_path), f"Features db path {db_path} does not exist"

        # Check the splits are not empty
        assert len(splits) > 0, "Splits cannot be empty"

        # Check the padding strategy is valid
        assert padding_strategy in [
            "zero",
            "mean",
        ], f"Padding strategy {padding_strategy} is not valid"

        if keywords == "default":
            keywords = DEFAULT_KEYWORDS
        elif not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list of strings or 'default'. Found {type(keywords)}"
            )

        # Check the given keywords are in the list of available DEFAULT_KEYWORDS
        for keyword in keywords:
            assert keyword in DEFAULT_KEYWORDS, f"Keyword {keyword} is not available"

        # Substract keywords from self.keywords
        if not_keywords is not None:
            keywords = [keyword for keyword in keywords if keyword not in not_keywords]

        # If means and stds are given, they should have the same length as keywords
        if features_means is not None:
            assert len(features_means) == len(
                keywords
            ), "Features means should have the same length as keywords"

        if features_stds is not None:
            assert len(features_stds) == len(
                keywords
            ), "Features stds should have the same length as keywords"

        # Must provide both means and stds or neither
        if (features_means is not None and features_stds is None) or (
            features_means is None and features_stds is not None
        ):
            raise ValueError(
                "Must provide both features means and features stds. Or provide neither"
            )

        # Must provide same keys
        if features_means is not None and features_stds is not None:
            assert set(features_means.keys()) == set(
                keywords
            ), "Features means should have the same keys as keywords"
            assert set(features_stds.keys()) == set(
                keywords
            ), "Features stds should have the same keys as keywords"

        # Set the attributes
        self.mode = mode
        self.T = T
        self.L = L
        self.S = S
        self.B = B
        self.MIN_FLARE_CLASS = MIN_FLARE_CLASS
        self.ALLOW_OVERLAPS = ALLOW_OVERLAPS
        self.db_path = db_path
        self.keywords = keywords
        self.padding_strategy = padding_strategy
        self.features_means = features_means
        self.features_stds = features_stds
        self.splits = splits
        self.use_only_active = use_only_active
        self.temp_db_id = temp_db_id
        self.use_PCA = use_PCA
        self.n_PCA_components = n_PCA_components
        self.pca_columns = None
        self.PCA = PCA_object
        self.use_is_valid = use_is_valid

        # temp db is in the folder of db_path but inside a tmp folder
        self.temp_db_root = os.path.join(os.path.dirname(db_path), "tmp")

        # Check if the temp db root exists
        if not os.path.exists(self.temp_db_root):
            os.makedirs(self.temp_db_root, exist_ok=True)

        self.temp_db_path = os.path.join(self.temp_db_root, f"{self.temp_db_id}.db")

        formatted_T = format_float_for_table_name(T)
        formatted_L = format_float_for_table_name(L)
        formatted_S = format_float_for_table_name(S)
        formatted_B = format_float_for_table_name(B)

        self.dataset_table_name = f"""T_{formatted_T}_L_{formatted_L}_S_{formatted_S}_B_{formatted_B}_OVERLAPS_{int(ALLOW_OVERLAPS)}_MIN_FLARE_{MIN_FLARE_CLASS}_dataset"""

        # Now we need a unique name for the temporary table.
        self.temp_table_name = self._create_unique_temp_table_name()

    def _create_unique_temp_table_name(self):
        name = f"temp_features_{self.mode}_{str(uuid.uuid4())}"

        # Can't have - in the name
        name = name.replace("-", "_")
        return name

    def _initialize_conn(self):
        "Establish connection to the database if not existent."
        if not hasattr(self, "conn") or self.conn is None:
            # Here the connection is not read-only as
            # we need to modify the database
            self.conn = sqlite3.connect(self.temp_db_path)

            # Attach the original database
            self.conn.execute(f"ATTACH DATABASE '{self.db_path}' AS base")

        # Otherwise the connection is already established

    def _setup_dataset(self):
        print("INITIALIZING CONNECTION")
        self._initialize_conn()
        print("COPYING DATASET TABLE")
        self._copy_dataset_table()  # Copy the dataset table
        print("CREATING TEMPORARY TABLE")
        self._create_temporary_table()  # Copy into temporary table the relevant data
        print("NORMALIZING FEATURES")
        self._normalize_features()  # Normalize to means 0 and std 1. Using given means and stds or database means and stds.
        print("APPLYING PADDING STRATEGY")
        self._apply_padding_strategy()  # Apply the padding strategy

        if self.use_PCA:
            # If we're doing PCA, then now is the time
            print("APPLYING PCA")
            self._apply_PCA()

        self.update_indices()

        print("CLOSE CONNECTION")
        self._close_connection()  # Close the connection

    def create_temp_table_for_pca(self):
        columns = [f"pca_component_{i}" for i in range(self.n_PCA_components)]
        self.pca_columns = columns

        cursor = self.conn.cursor()

        # Create temp_pca_data
        query = """
        CREATE TEMP TABLE temp_pca_data (
        harpnum INT,
        timestamp TEXT,
        """

        for column in self.pca_columns:
            query = query + f"{column} REAL,"

        query = query[:-1] + ")"

        cursor.execute(query)

        cursor.close()
        self.conn.commit()

    def insert_transformed_data(self, ids, transformed_data):
        placeholders = ", ".join(
            ["?"] * (self.n_PCA_components + 2)
        )  # +2 for the harpnum, timestamp
        print("IDS:", ids[:10])
        with self.conn as conn:
            conn.executemany(
                f"INSERT INTO temp_pca_data VALUES ({placeholders})",
                [
                    tuple([int(id[0])] + [str(id[1])] + list(components))
                    for id, components in zip(ids, transformed_data)
                ],
            )

        conn.commit()

    def update_original_table(self):
        cur = self.conn.cursor()

        cur.execute(
            f"""
        CREATE TABLE {self.temp_table_name}_full AS SELECT
        TT.*, TPD.* FROM {self.temp_table_name} TT
        INNER JOIN temp_pca_data TPD
        ON TT.harpnum = TPD.harpnum AND TT.Timestamp = TPD.timestamp
        """
        )

        # Make sure same number of rows

        original_n_rows = cur.execute(
            f"SELECT COUNT(*) FROM {self.temp_table_name}"
        ).fetchall()[0]
        updated_n_rows = cur.execute(
            f"SELECT COUNT(*) FROM {self.temp_table_name}_full"
        ).fetchall()[0]

        assert original_n_rows == updated_n_rows

        # Now delete the original table_exists
        cur.execute(f"DROP TABLE {self.temp_table_name}")

        # And rename the full one
        cur.execute(
            f"ALTER TABLE {self.temp_table_name}_full RENAME TO {self.temp_table_name}"
        )

        cur.close()
        self.conn.commit()

    def delete_temp_table_for_pca(self):
        with self.conn as conn:
            conn.execute("DROP TABLE IF EXISTS temp_pca_data")

    def _apply_PCA(self):
        # Fetch data from the original table
        data = np.array(
            self.conn.execute(
                f"SELECT harpnum, Timestamp, {', '.join(self.keywords)} FROM {self.temp_table_name}"
            ).fetchall()
        )
        harpnum, timestamp = data[:, 0].astype(int), data[:, 1].astype(str)
        features = data[:, 2:].astype(float)

        print("harpnums:", harpnum[:10])
        print("timestamps:", timestamp[:10])

        # Apply PCA if not already provided
        if self.PCA is None:
            self.PCA = cuPCA(n_components=self.n_PCA_components)
            self.PCA.fit(features)
            print(f"Explained variance: {self.PCA.explained_variance_ratio_.sum()}")

        transformed_data = self.PCA.transform(features)

        ids = np.concatenate((harpnum[:, np.newaxis], timestamp[:, np.newaxis]), axis=1)

        # Execute the PCA data handling workflow
        self.create_temp_table_for_pca()
        self.insert_transformed_data(ids, transformed_data)
        self.update_original_table()
        self.delete_temp_table_for_pca()

    def _close_connection(self):
        """
        Closes the connection to the database
        """
        self.conn.close()
        delattr(self, "conn")

    def _copy_dataset_table(self):
        # Check if the dataset table exists
        cur = self.conn.cursor()

        cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.dataset_table_name}'"
        )

        table_exists = len(cur.fetchall()) > 0

        if not table_exists:
            # We need to copy it
            cur.execute(
                f"CREATE TABLE {self.dataset_table_name} AS SELECT * FROM base.{self.dataset_table_name}"
            )

            cur.execute(
                f"CREATE TABLE {self.dataset_table_name}_splits AS SELECT * FROM base.{self.dataset_table_name}_splits"
            )

        return

    def _create_temporary_table(self):
        """
        Create a copy of the padded features into a unique "temporary" table

        ONLY TO BE CALLED BY _setup_dataset
        """
        cur = self.conn.cursor()

        # We create a temp_features table
        # Not sure but this may fail as I'm not adding PF. before keywords
        query = ""

        if self.use_only_active:
            query = f"""
            CREATE TABLE IF NOT EXISTS {self.temp_table_name} AS SELECT
            PF.harpnum, PF.Timestamp, DTS.split, PF.IS_VALID, PF.IS_TMFI, {','.join(self.keywords)}
            FROM base.padded_features PF
            INNER JOIN base.{self.dataset_table_name}_splits DTS ON PF.harpnum = DTS.harpnum
            WHERE DTS.split IN ({','.join([str(x) for x in self.splits])}) AND DTS.is_active = 1
            ORDER BY PF.harpnum, PF.Timestamp ASC
            """
        else:
            query = f"""
            CREATE TABLE IF NOT EXISTS {self.temp_table_name} AS SELECT
            PF.harpnum, PF.Timestamp, DTS.split, PF.IS_VALID, PF.IS_TMFI, {','.join(self.keywords)}
            FROM base.padded_features PF
            INNER JOIN base.{self.dataset_table_name}_splits DTS ON PF.harpnum = DTS.harpnum
            WHERE DTS.split IN ({','.join([str(x) for x in self.splits])})
            ORDER BY PF.harpnum, PF.Timestamp ASC
            """

        cur.execute(query)

        # Index on harpnum, timestamp
        cur.execute(
            f"""
        CREATE INDEX IF NOT EXISTS {self.temp_table_name}_harpnum_timestamp_index ON {self.temp_table_name}(harpnum, Timestamp)
        """
        )

        # Index on IS_VALID
        cur.execute(
            f"""
        CREATE INDEX IF NOT EXISTS {self.temp_table_name}_is_valid_index ON {self.temp_table_name}(is_valid)
        """
        )

        cur.close()
        self.conn.commit()

    def update_indices(self):
        cur = self.conn.cursor()
        # Index on harpnum, timestamp
        cur.execute(
            f"""
        CREATE INDEX IF NOT EXISTS {self.temp_table_name}_harpnum_timestamp_index ON {self.temp_table_name}(harpnum, Timestamp)
        """
        )

        # Index on IS_VALID
        cur.execute(
            f"""
        CREATE INDEX IF NOT EXISTS {self.temp_table_name}_is_valid_index ON {self.temp_table_name}(is_valid)
        """
        )

        cur.close()
        self.conn.commit()

    def _apply_padding_strategy(self):
        "Apply the padding strategy to the data"

        # Get the cursor
        cur = self.conn.cursor()

        if self.padding_strategy == "zero":
            # Set every keyword where IS_VALID is 0 to 0
            cur.execute(
                f"""
                UPDATE {self.temp_table_name}
                SET {', '.join([f"{keyword} = 0" for keyword in self.keywords])}
                WHERE IS_VALID = 0
                """
            )
        elif self.padding_strategy == "mean":
            # Set every keyword where IS_VALID is 0 to the mean
            for keyword in self.keywords:
                cur.execute(
                    f"""
                UPDATE {self.temp_table_name}
                SET {', '.join([f"{keyword} = {self.calculated_features_means[keyword]}" for keyword in self.keywords])}
                WHERE IS_VALID = 0
                """
                )

        cur.close()
        self.conn.commit()

    def _normalize_features(self):
        """
        Normalize the features to have mean 0 and std 1

        TO BE CALLED BY _setup_dataset AND AFTER _create_temporary_table
        ACTS ON the "temporary" features table
        """
        self.calculated_features_means = {}
        self.calculated_features_stds = {}

        cur = self.conn.cursor()
        # We normalize the features
        for keyword in self.keywords:
            # Get the mean and std
            # If no means and stds were given, use the ones from the database
            if self.features_means is not None and self.features_stds is not None:
                mean, std = self.features_means[keyword], self.features_stds[keyword]
            else:
                # Calculate mean
                cur.execute(
                    f"SELECT AVG({keyword}) FROM {self.temp_table_name} WHERE IS_VALID = 1"
                )
                mean = cur.fetchone()[0]

                # Calculate standard deviation manually
                cur.execute(
                    f"""
                    SELECT SQRT(SUM(({keyword} - {mean}) * ({keyword} - {mean})) / COUNT({keyword}))
                    FROM {self.temp_table_name}
                    WHERE IS_VALID = 1
                    """
                )
                std = cur.fetchone()[0]

            # Add to the calculated means and stds
            self.calculated_features_means[keyword] = mean
            self.calculated_features_stds[keyword] = std

        # Update the table
        cur.execute(
            f"""
        UPDATE {self.temp_table_name}
        SET {', '.join(
        [
        f"{keyword} = ({keyword} - {self.calculated_features_means[keyword]}) / {self.calculated_features_stds[keyword]}" for keyword in self.keywords
        ]
        )}
        """
        )

        # Close the cursor
        cur.close()
        self.conn.commit()

    def wait_for_cleanup(self):
        """
        Waits for the cleanup event to be set and then
        deletes the temporary table
        """
        cleanup_event.wait()
        self.cleanup()

    def cleanup(self):
        """
        Deletes the temporary table
        """
        print("Cleaning up temporary table")
        self._initialize_conn()
        cur = self.conn.cursor()

        cur.execute(f"DROP TABLE {self.temp_table_name}")

        cur.close()
        self.conn.commit()

        # Close the connection too
        self.conn.close()
        delattr(self, "conn")


class BaseSWANDataset(BaseDataset):
    def __init__(
        self,
        mode: str,  # Mode of the dataset. Either train or validation (which includes testing)
        db_path: str,  # Path to features db
        splits: list,  # List of splits to use in the databse (for cross-validation and splitting into training/test)
        temp_table_name: str,  # Name of the temporary table
        T: float,  # Flare happens within next T hours. Does it have CME?
        L: float,  # We take observations of length L hours so end of observation within T to flare
        S: float,  # Observations are S hours apart
        B: float,  # We don't allow flares within observations. We also consider B hours of buffer after the flare as non ok.
        C: float,  # This is the cadence, so data points are taken over a period of L hours with each separated by C hours
        MIN_FLARE_CLASS: float,  # Only flares of class larger than this. For example, 35 is M5
        ALLOW_OVERLAPS: bool,  # If this is false, we allow flares in the observation period
        use_only_active: bool = False,  # If true, only use regions that every produced a flare
        keywords: list = DEFAULT_KEYWORDS,  # List of keywords to use. DEFAULT_KEYWORDS is a list of all available keywords
        not_keywords: list = None,  # List of keywords to not use. If None, all keywords are used.
        noise_level: float = 0.05,
        use_PCA: bool = False,
        use_is_valid: bool = False,
        *args,  # Any other
        **kwargs,  # Any other
    ):
        self._initialize_parameters(
            mode,
            db_path,
            splits,
            temp_table_name,
            T,
            L,
            S,
            B,
            C,
            MIN_FLARE_CLASS,
            ALLOW_OVERLAPS,
            keywords,
            not_keywords,
            noise_level,
            use_only_active,
            use_PCA,
            use_is_valid,
        )

    def _initialize_parameters(
        self,
        mode: str,
        db_path: str,
        splits: list,
        temp_table_name: str,
        T: float,
        L: float,
        S: float,
        B: float,
        C: float,
        MIN_FLARE_CLASS: float,
        ALLOW_OVERLAPS: bool,
        keywords: list,
        not_keywords: list,
        noise_level: float,
        use_only_active: bool,
        use_PCA: bool,
        use_is_valid: bool,
    ):
        """
        Initialize the parameters of the dataset.
        """
        # Mode should either be train or val
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Mode {mode} is not valid. Must be either train, val or test"

        # Check T, L, S and B are positive
        assert T > 0, "T must be > 0"
        assert L > 0, "L must be > 0"
        assert S > 0, "S must be > 0"
        assert B >= 0, "B must be >= 0"
        assert MIN_FLARE_CLASS > 0, "MIN_FLARE_CLASS must be positive"

        # Cadence is a bit more involved, as it must be a multiple of 12 minutes
        # Therefore, in hours it must be a multiple of 0.2

        assert C > 0, "C must be > 0"
        assert (
            C % 0.2 == 0
        ), "C must be a multiple of 0.2. This is because the data is taken every 12 minutes"

        # And now IMPORTANT. We store C in indices, not in hours
        # so we can then use it to index the data
        self.C = int(C / 0.2)

        if not use_PCA:
            if keywords == "default":
                keywords = DEFAULT_KEYWORDS
            elif not isinstance(keywords, list):
                raise ValueError(
                    f"Keywords must be a list of strings or 'default'. Found {type(keywords)}"
                )

            # Check the given keywords are in the list of available DEFAULT_KEYWORDS
            for keyword in keywords:
                assert (
                    keyword in DEFAULT_KEYWORDS
                ), f"Keyword {keyword} is not available"

            # Substract keywords from self.keywords
            if not_keywords is not None:
                keywords = [
                    keyword for keyword in keywords if keyword not in not_keywords
                ]

        # With this we can calculate the expected shape of the data
        self.use_is_valid = use_is_valid

        self.output_shape = (int(L / C), len(keywords) + int(self.use_is_valid))

        # Check db path exists
        assert os.path.exists(db_path), f"Features db path {db_path} does not exist"

        # Check the splits are not empty
        assert len(splits) > 0, "Splits cannot be empty"

        # Check noise level is between 0 and 1
        assert (
            noise_level >= 0 and noise_level <= 1
        ), f"Noise level must be between 0 and 1. Found {noise_level}"

        # Set the attributes
        self.mode = mode
        self.T = T
        self.L = L
        self.S = S
        self.B = B
        self.MIN_FLARE_CLASS = MIN_FLARE_CLASS
        self.ALLOW_OVERLAPS = ALLOW_OVERLAPS
        self.db_path = db_path
        self.splits = splits
        self.keywords = keywords
        self.noise_level = noise_level
        self.keyword_str = ",".join(self.keywords)
        self.use_only_active = use_only_active

        formatted_T = format_float_for_table_name(T)
        formatted_L = format_float_for_table_name(L)
        formatted_S = format_float_for_table_name(S)
        formatted_B = format_float_for_table_name(B)

        self.dataset_table_name = f"""T_{formatted_T}_L_{formatted_L}_S_{formatted_S}_B_{formatted_B}_OVERLAPS_{int(ALLOW_OVERLAPS)}_MIN_FLARE_{MIN_FLARE_CLASS}_dataset"""

        self.temp_table_name = temp_table_name

    def _initialize_conn(self):
        "Establish connection to the database if not existent."
        if not hasattr(self, "conn"):
            # Use uri of style "'file:/path/to/database?mode=ro'" to open in read only mode
            uri = f"file:{self.db_path}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)

            # Check that the dataset table exists
            self._check_dataset_tables_exist()

            # Store the periods in memory
            # This also calls inside _process_periods
            self._periods = self._get_periods()

            # Need to also get the positive weight
            self._pos_weight = self._get_pos_weight()

        # Otherwise the connection is already established

    def _get_pos_weight(self):
        return NotImplementedError

    def _process_periods(self, periods) -> pd.DataFrame:
        return periods

    def _get_periods(self) -> np.ndarray:
        """
        Reads the dataset periods into memory
        """
        query = f"""
        SELECT 
        MDT.harpnum, -- 0
        MDT.period_start, -- 1 
        MDT.period_end,  -- 2
        MDT.activity_in_obs, -- 3
        MDT.has_flare_above_threshold, -- 4
        MDT.has_flare_below_threshold, -- 5
        MDT.flare_above_threshold_id, -- 6
        MDT.flare_below_threshold_id, -- 7
        MDT.has_cme_flare_below_threshold, -- 8
        MDT.has_cme_flare_above_threshold, -- 9
        MDT.has_cme_no_flare, -- 10
        MDT.cme_flare_above_threshold_id, -- 11
        MDT.cme_flare_below_threshold_id, -- 12
        MDT.cme_no_flare_id, -- 13
        DTS.is_active, -- 14
        DTS.is_cme_active -- 15
        FROM {self.dataset_table_name} MDT
        INNER JOIN {self.dataset_table_name}_splits DTS
        ON DTS.harpnum = MDT.harpnum
        WHERE DTS.split IN ({','.join([str(x) for x in self.splits])})
        """

        if self.use_only_active:
            query += "AND DTS.is_active = 1"

        periods = pd.read_sql(
            query,
            self.conn,
        )

        periods = self._process_periods(periods)

        periods = periods.to_numpy()

        return periods

    def _get_all_labels(self) -> np.ndarray:
        raise NotImplementedError

    def _get_baseline_predictions(self) -> np.ndarray:
        raise NotImplementedError

    def get_baseline_metrics(self) -> dict:
        y_true = self._get_all_labels()
        y_pred = (
            self._get_baseline_predictions()
        )  # self._periods[:, 14].astype(int)  # is_active
        print(y_true, y_pred)
        cf = confusion_matrix(y_true, y_pred)

        # Unravel the cf
        tn, fp, fn, tp = cf.ravel()

        tss = tp / (tp + fn) - fp / (fp + tn)

        return {
            "baseline_TSS": float(tss),
            "baseline_average_precision_score": float(
                average_precision_score(y_true, y_pred)
            ),
        }

    def _check_dataset_tables_exist(self):
        """
        Check that the required table exists in the database.
        """
        cur = self.conn.cursor()
        dataset_exists = cur.execute(
            f"""
        SELECT name FROM sqlite_master WHERE type='table' AND name='{self.dataset_table_name}'
        """
        ).fetchone()

        temp_table_exists = cur.execute(
            f"""
        SELECT name FROM sqlite_master WHERE type='table' AND name='{self.temp_table_name}'
        """
        )

        if dataset_exists is None:
            raise ValueError(
                f"Dataset table {self.dataset_table_name} does not exist in the database"
            )

        if temp_table_exists is None:
            raise ValueError(
                f"Temporary table {self.temp_table_name} does not exist in the database"
            )

    def __del__(self):
        "Close the connection to the database if it exists."
        if hasattr(self, "conn"):
            self.conn.close()

    def _get_data(self, harpnum: int, start: str, end: str):
        """
        Retrieves the data for a particular row
        """
        cur = self.conn.cursor()

        query = f"""
        SELECT {','.join(["IS_VALID"] + self.keywords)} FROM {self.temp_table_name}
        WHERE harpnum = ?
        AND Timestamp BETWEEN ? AND ?
        """

        data = cur.execute(query, (int(harpnum), start, end)).fetchall()

        cur.close()

        # We sample at cadence C

        data = data[:: self.C]

        # Now this is a list of lists. We want a tensor with output_shape
        # (int(L/C), len(keywords))

        data = torch.tensor(data, dtype=torch.float32)

        is_valid_mask = data[:, 0].bool()
        data = data[:, 1:].float()

        # Now ensure it's of the right shape
        assert (
            data.shape == self.output_shape
        ), f"HARPNUM: {harpnum}, start {start}, end {end}:\nData shape {data.shape} is not the expected {self.output_shape}"

        return data, is_valid_mask

    def _get_label(self, row) -> int:
        raise NotImplementedError

    def _get_base_metadata(self, row) -> dict:
        metadata = {
            "harpnum": row[0],
            "start_date": row[1],
            "end_date": row[2],
            "activity_in_obs": row[3],
            "has_cme_flare_above_threshold": row[9],
            "has_cme_flare_below_threshold": row[8],
            "has_cme_no_flare": row[10],
            "has_flare_above_threshold": row[4],
            "has_flare_below_threshold": row[5],
        }
        return metadata

    def _get_metadata(self, row) -> dict:
        raise NotImplementedError

    def _add_gaussian_white_noise(self, data):
        noise = torch.randn(data.shape) * self.noise_level

        return data + noise

    def __getitem__(self, idx):
        self._initialize_conn()
        # Get the row at idx in the dataset table from those with split in splits

        row = self._periods[idx]

        harpnum = row[0]
        start = row[1]
        end = row[2]

        data, is_valid_mask = self._get_data(harpnum, start, end)

        # Only add noise for training
        if self.mode == "train":
            data = self._add_gaussian_white_noise(data)

        label = self._get_label(row)

        metadata = self._get_metadata(row)

        metadata["IS_VALID"] = is_valid_mask

        return data, label, metadata

    def __len__(self):
        self._initialize_conn()
        return len(self._periods)
