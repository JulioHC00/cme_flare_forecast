import os
import sqlite3
import pandas as pd
import torch
import numba
import numpy as np
from copy import deepcopy
import zarr
from dataclasses import dataclass
from datetime import datetime
from torch import tensor as tt
from abc import abstractmethod
from omegaconf import DictConfig
from .base_dataset import BaseDataset
from ..transforms.utils.transforms_factory import transforms_factory


@numba.njit()
def smooth_label_function_vectorized(x, H, THETA, D, EPSILON):
    """
    Defines the smooth label function.

    The function is

    l(x) =

    e^((x^2 / H^2) * ln(THETA)) if x <= 0
    e^((x^2 / D^2) * ln(EPSILON)) if x > 0

    Values smaller than EPSILON are set to 0, max value is 1.

    Where:
        x: Time until next event, measured as current time - next event time
        so negative for times before event and positive for times after event.
        H: the forecast horizon
        THETA: the threshold for the forecast horizon. This means that the point (H, THETA)
        is in the function. The idea is, by using a threshold THETA, binary labels for
        the forecast horizon H are recovered.
        D: Decay width, measures how quickly the function returns to 0 after the event. At
        value D, the funciton reaches EPSILON. After that, the values are set to 0.
        EPSILON: A small number that represents the smallest label we use before setting
        values to 0. It also affects how quick the decay is.

    Comments:
        Values of x smaller than GAMMA = H * sqrt(ln(EPSILON) / ln(THETA)) will always be 0, because
        l(GAMMA) = EPSILON and l(x < GAMMA) < EPSILON. So, GAMMA represents when the function starts
        rising. Also, values of x larger than D will also be 0. So D measures when the function
        returns to absolute 0.
    """
    H2_log_theta = np.log(THETA) / H**2  # Precompute constant for x <= 0
    D2_log_epsilon = np.log(EPSILON) / D**2  # Precompute constant for x > 0

    rows, cols = x.shape
    result = np.empty_like(x)  # Initialize an empty array of the same shape as x

    for i in range(rows):
        for j in range(cols):
            xi = x[i, j]
            if xi <= 0:
                value = np.exp(xi**2 * H2_log_theta)
            else:
                value = np.exp(xi**2 * D2_log_epsilon)

            result[i, j] = value if value >= EPSILON else 0

    return result


# Numba optimized functions that can't be inside the class
@numba.njit(parallel=True)
def cme_history_function_vectorized(
    x: np.ndarray, TAU: float, EPSILON: float
) -> np.ndarray:
    """
    Vectorized function to calculate the history values based on the time differences.

    This function applies a transformation to each element in the input array `x`.
    The transformation is based on the exponential decay model, where values are
    calculated as zero for non-positive inputs and as an exponential decay function
    for positive inputs.

    Parameters:
    - x (np.array): An array of time differences (in hours) between the observation
    time and CME times.

    Returns:
    - np.array: An array of history values calculated for each element in `x`.

    Notes:
    - The calculation uses the TAU and EPSILON parameters from `self.history_params`.
    TAU is used as the decay constant, and EPSILON is the base of the exponential decay.
    """
    results = np.exp((x**2 / TAU**2) * np.log(EPSILON))

    return results


# Create data class for the smooth label parameters


@dataclass
class SmoothLabelParams:
    def __init__(self, H, THETA, D, EPSILON):
        self.H = H
        self.THETA = THETA
        self.D = D
        self.EPSILON = EPSILON

    def validate_args(self):
        """
        Validates the parameters of the class instance to ensure they meet specific criteria.
        Raises:
            ValueError: If `H` is not above 0.
            ValueError: If `THETA` is not above 0 and below 1.
            ValueError: If `D` is not above 0.
            ValueError: If `EPSILON` is not above 0 and smaller than 1e-4
        """
        # Ensure H is above 0
        if self.H <= 0:
            raise ValueError(f"H must be above 0, got {self.H}")
        # Ensure THETA is above 0
        if not 0 < self.THETA < 1:
            raise ValueError(f"THETA must be in the range (0, 1), got {self.THETA}")
        # Ensure D is above 0
        if self.D <= 0:
            raise ValueError(f"D must be above 0, got {self.D}")
        # Ensure EPSILON is above 0
        if 0 < self.EPSILON < 1e-4:
            raise ValueError(f"EPSILON must be in range (0, 1e-4), got {self.EPSILON}")


@dataclass
class HistoryParams:
    def __init__(self, TAU, EPSILON):
        self.TAU = TAU
        self.EPSILON = EPSILON

    def validate_args(self):
        """
        Validates the parameters of the class instance to ensure they meet specific criteria.
        Raises:
            ValueError: If `TAU` is not above 0.
            ValueError: If `EPSILON` is not above 0 and smaller than 1e-3
        """
        # Ensure TAU is above 0
        if not self.TAU > 0:
            raise ValueError(f"THETA must be in the range (0, 1), got {self.TAU}")
        # Ensure EPSILON is above 0
        # NOTE: For the labels we required max value to be 1e-4, here it's 1e-3
        if 0 < self.EPSILON < 1e-3:
            raise ValueError(f"EPSILON must be in range (0, 1e-3), got {self.EPSILON}")


# To be used at some point in the future when I feel like adding the flares table
# because right now it's not there and I can't be bothered to add it
class FlareClassTranslator:
    def __init__(self):
        self.flare_classes = {
            "A": 0,
            "B": 1,
            "C": 2,
            "M": 3,
            "X": 4,
        }

    def __call__(self, flare_class: str):
        """
        Translates a solar flare class (e.g., "M1.2") into a numerical representation.
        The letter indicates the primary class, while the following number represents a subclass.
        """
        if len(flare_class) == 1:
            if (
                not flare_class.isalpha()
                or flare_class not in self.flare_classes.keys()
            ):
                raise ValueError(f"Invalid flare class format: {flare_class}")
            else:
                return self.flare_classes[flare_class]

        if (
            not flare_class[0].isalpha()
            or not flare_class[1:].replace(".", "", 1).isdigit()
        ):
            raise ValueError(f"Invalid flare class format: {flare_class}")

        fclass = flare_class[0].upper()
        if fclass not in self.flare_classes.keys():
            raise ValueError(f"Invalid flare class: {flare_class}")

        subclass = float(flare_class[1:]) * 0.1
        return self.flare_classes[fclass] + subclass


class BaseSDOMLDataset(BaseDataset):
    VALID_MODES = ["train", "val", "test"]

    def __init__(
        self,
        db_path: str,
        db_main_table: str,
        db_harps_table: str,
        db_groups_table: str,
        data_path: str,
        splits: list,
        subsplits: list,
        dtype: str,
        device: str,
        transforms_config: DictConfig,
        mode: str = "train",
        smooth_label_params: dict = {"H": 24, "THETA": 0.5, "D": 6, "EPSILON": 1e-6},
        history_params: dict = {"TAU": 72, "EPSILON": 1e-6},
    ):
        """
        Initialize the dataset.

        Parameters:
        - db_path (str): The path to the database file.
        - data_path (str): The path to the data directory.
        - splits (list): A list of splits to use. If None, use all splits.
        - subsplits (list): A list of subsplits to use. If None, use all subsplits.
        - dtype (str): The dtype to use for the data.
        - device (str): The device to use for the data.
        - transforms_config (DictConfig): The transforms configuration.
        - mode (str): The mode to use. Can be "train", "val" or "test".
        - smooth_label_params: dict = {"H": 24, "THETA": 0.5, "D": 6, "EPSILON": 1e-6},
        """
        super(BaseSDOMLDataset, self).__init__()
        # Set all attributes
        self.db_path = db_path
        self.db_main_table = db_main_table
        self.db_harps_table = db_harps_table
        self.db_groups_table = db_groups_table
        self.db_table = "COLLATED_TABLE"
        self.data_path = data_path
        self.transforms_config = transforms_config
        self.mode = mode
        self.dtype = getattr(torch, dtype)
        self.device = device
        self.splits = splits
        self.subsplits = subsplits
        self.smooth_label_params: SmoothLabelParams = SmoothLabelParams(
            **smooth_label_params
        )
        self.history_params: HistoryParams = HistoryParams(**history_params)

        # Validate the arguments
        self.validate_args()

        # Load the database
        self.load_db()

        # Now load the metadata and pre-load the data
        self.init_data()

        # Now calculate the weight dictionaries
        self.init_weights()

        # Get the transforms
        self.init_transforms()

        # And close the database
        self.close_db()

    def validate_args(self):
        """
        Validates the parameters of the class instance to ensure they meet specific criteria.

        Raises:
            ValueError: If `db_path` does not exist.
            ValueError: If `data_path` does not exist.
            ValueError: If `mode` is not one of 'train', 'val', or 'test'.

        Note:
            Additional parameters may be added in the future, requiring updates to this validation method.
        """
        # Ensure that the db_path is valid
        if not os.path.exists(self.db_path):
            raise ValueError("The database file does not exist.")

        # Ensure data path exists
        if not os.path.exists(self.data_path):
            raise ValueError("The data directory does not exist.")

        # Ensure the mode is valid
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {self.mode}")

    def init_data(self):
        """
        Initializes the data-related attributes of the class instance.

        Attributes:
            dataset: Dataframe containing the dataset.
            zarrs: Dictionary of pre-loaded zarrs. Keys are the harpums

        Calls:
            load_dataset(): Method to load and return metadata.
            preload_zarr(): Method to preload and return Zarr data.
        """

        self.harps_cmes: dict = self.get_harps_cmes()

        self.dataset: pd.DataFrame = self.load_dataset()

        self.zarrs: dict = self.preload_zarr()

    def init_weights(self):
        # Not being assigned to anything as they are stored directly in the df.
        # Get weights for the data
        self.get_loss_weights()

        # Get sampling weights
        self.get_sampling_weights()

    def load_db(self):
        """
        Loads the database and initializes relevant database-related attributes.

        Attributes:
            conn: SQLite database connection object.
            cursor: SQLite database cursor object for executing SQL queries.

        Executes:
            A SQL query to create a temporary collated table by joining multiple tables.

        Raises:
            sqlite3.OperationalError: If the SQL query fails for any reason.
        """

        # Connect to the database
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        # Now we need to create a collated table

        self.cursor.execute(
            f"""
        CREATE TEMPORARY TABLE IF NOT EXISTS {self.db_table} AS
        SELECT M.*, G.split, G.subsplit FROM {self.db_main_table} M
        INNER JOIN {self.db_harps_table} H ON M.harpnum = H.harpnum
        INNER JOIN {self.db_groups_table} G ON H.group_id = G.group_id
        """
        )

    def close_db(self) -> None:
        """
        Close the database.

        Also deletes the connection and cursor objects to avoid pickling errors.
        """
        self.conn.close()

        # Need to delete the connection and cursor because
        # otherwise I get an error when pickling the dataset

        del self.conn
        del self.cursor

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the metadata from the database for the specified splits and subsplits.

        Returns:
            pd.DataFrame: The loaded dataset containing metadata.

        Attributes Modified:
            None

        Notes:
            - `splits` and `subsplits` should be valid attributes of the class instance.
            - Replaces null values in 'verification_level' with 0.
            - Adds a 'below_cap' column to indicate if the examples are below the defined cap.

        Raises:
            ValueError: If no data is found for the given splits and subsplits.
            ValueError: If no examples are found below the cap.
        """
        # Convert splits and subsplits to string format for SQL query
        splits_str: str = ", ".join([f"'{split}'" for split in self.splits])
        subsplits_str: str = ", ".join([f"'{subsplit}'" for subsplit in self.subsplits])

        # Get the metadata from the database
        dataset: pd.DataFrame = pd.read_sql_query(
            f"SELECT * FROM {self.db_table} WHERE split IN ({splits_str}) AND subsplit IN ({subsplits_str})",
            self.conn,
        )

        # Replace null verification levels with 0
        dataset["verification_level"] = dataset["verification_level"].fillna(0)

        # Convert end_image to datetime
        dataset["end_image_seconds"] = (
            pd.to_datetime(dataset["end_image"]).astype("int64") // 1e9
        )

        harpnums, timestamps = (
            dataset["harpnum"].to_numpy(),
            dataset["end_image_seconds"].to_numpy(),
        )

        # Vectorized datetime calculations
        cme_times_uneven = [
            self.harps_cmes[HARPNUM] if HARPNUM in self.harps_cmes else []
            for HARPNUM in harpnums
        ]

        max_length = max(map(len, cme_times_uneven))

        CME_TIMES_SECONDS = np.full((len(cme_times_uneven), max_length), np.nan)

        # Fill the array using slicing
        for i, row in enumerate(cme_times_uneven):
            CME_TIMES_SECONDS[i, : len(row)] = row

        CME_HOUR_DIFFS = (timestamps[:, np.newaxis] - CME_TIMES_SECONDS) / 3600
        self.hours_diff = CME_HOUR_DIFFS

        # Calculate smooth probabilistic labels
        dataset["smooth_label"] = self.get_smooth_label(deepcopy(CME_HOUR_DIFFS))

        # Calculate the cme history
        dataset["cme_history"] = self.get_cme_history(deepcopy(CME_HOUR_DIFFS))

        # Validate the loaded dataset
        # 1. Check if there's data for the given splits and subsplits
        if len(dataset) == 0:
            raise ValueError("No data found for the given splits and subsplits")

        # 2. Check if there are examples above the threshold THETA
        if sum(dataset["smooth_label"] > self.smooth_label_params.THETA) == 0:
            # The reason to do this is that in principle, the point (H,THETA) represents
            # our forecast horizon and the threshold THETA recovers binary labels so we
            # want to have some examples above that threshold
            raise ValueError(
                f"No single label is above THETA={self.smooth_label_params.THETA}"
            )

        # 3. Check no values smaller than EPSILON (by definition we set them to 0)
        if np.any(
            dataset["smooth_label"].between(
                0, self.smooth_label_params.EPSILON, inclusive="neither"
            )
        ):
            raise ValueError(
                f"Values smaller than EPSILON={self.smooth_label_params.EPSILON} found"
            )

        return dataset

    def get_harps_cmes(self) -> dict:
        """
        Get the CME times per harpnum, using the FINAL_CME_HARPS_ASSOCIATIONS table.

        Now, it's true that not all CMEs in that table will be in the final dataset
        as some may e.g. have happened before the region was within the 70 degrees
        of longitude. Regardless, since the use of this methods is to get a smooth
        probabilistic label for regression, the idea of including all CMEs is valid.

        Returns:
            dict: A dictionary containing the CME times per harpnum
        """
        # Get the CME times per harpnum
        harps_cmes = pd.read_sql_query(
            """
        SELECT FCHA.harpnum, strftime('%s', C.cme_date) as cme_date FROM FINAL_CME_HARP_ASSOCIATIONS FCHA
        INNER JOIN CMES C ON FCHA.cme_id = C.cme_id
        """,
            self.conn,
        )

        harps_cmes["cme_date"] = harps_cmes["cme_date"].astype(int)

        # Group by harpnum
        harps_cmes = harps_cmes.groupby("harpnum")["cme_date"].apply(list).to_dict()

        return harps_cmes

    def preload_zarr(self) -> dict:
        """
        Preload zarr data from the data directory based on unique harp numbers in the dataset.

        Returns:
            dict: A dictionary containing zarr data, keyed by harpnum.

        Attributes Modified:
            None

        Notes:
            - `dataset` and `data_path` should be valid attributes of the class instance.
            - Assumes the zarr files are named after their respective harp numbers.
        """

        # Initialize an empty dictionary to store zarr data
        data: dict = dict()

        # Extract unique harp numbers from the dataset metadata
        harpnums = self.dataset["harpnum"].unique()

        # Loop through each unique harp number to load its zarr data
        for harpnum in harpnums:
            zarr_harpnum = zarr.open(
                os.path.join(self.data_path, f"{harpnum}"), mode="r"
            )

            # Store the loaded zarr data in the dictionary with harpnum as the key
            data[harpnum] = zarr_harpnum

        return data

    def get_imbalance_ratio(self) -> float:
        negative = sum(self.dataset["smooth_label"] == 0)
        positive = sum(self.dataset["smooth_label"] > 0)
        imbalance_ratio = negative / positive
        return imbalance_ratio

    def get_sampling_weights(self) -> list:
        """
        Calculates the sampling weights.

        Using the imbalance ratio between examples with prob > 0 and those with prob = 0,
        examples with prob > 0 are imbanced_ratio times more likely to be sampled.
        """
        # Weights depend on whether the sample is above or below cap
        # We first want the imbalance ratio
        imbalance_ratio = self.get_imbalance_ratio()

        # Vectorized computation of loss weights
        self.dataset["sampling_weight"] = np.where(
            self.dataset["smooth_label"] > 0, imbalance_ratio, 1
        )

        return self.dataset["sampling_weight"].tolist()

    def get_dataloader_weights(self):
        """
        Needed to conform with BaseDataset interface.

        Just returns the sampling weights.
        """
        return self.dataset["sampling_weight"].tolist()

    def get_loss_weights(self):
        """
        Get the loss weights dictionary based on the loss weight mode

        Loss weights are based on the imbalance ratio. A positive event is
        penalyzed by a factor of imbalance_ratio more than a negative event.
        """
        # We first want the imbalance ratio
        imbalance_ratio = self.get_imbalance_ratio()

        # Vectorized computation of loss weights
        self.dataset["loss_weight"] = np.where(
            self.dataset["smooth_label"] > 0, imbalance_ratio, 1
        )

        return self.dataset["loss_weight"].tolist()

    def init_transforms(self):
        self.transforms = transforms_factory(self.transforms_config, self.device)

    def get_cme_history(self, HOUR_DIFFS):
        """
        Calculates the cumulative history value for a given observation based on its
        proximity to previous Coronal Mass Ejections (CMEs).

        This method computes the sum of the history values for each CME event related
        to a specific solar region (identified by HARPNUM), based on the time difference
        between the observation time and the CME event times.

        Parameters:
        - row (pd.Series): A row from the dataset containing at least 'harpnum' and
        'end_image' fields. 'harpnum' identifies the solar region, and 'end_image'
        is the timestamp of the observation.

        Returns:
        - float: The cumulative history value for the given observation, indicating its
        temporal relation to past CME events.

        Notes:
        - This method utilizes the `cme_history_function_vectorized` for efficient
        calculation.
        - The method returns 0 if there are no CME times associated with the given HARPNUM.
        - The method does not impose an upper cap on the cumulative history value.
        """
        # Extract TAU and EPSILON once
        TAU = self.history_params.TAU
        EPSILON = self.history_params.EPSILON

        # Anything that is less thant 0 or larger than TAU is set to nan
        mask = (HOUR_DIFFS < 0) | (HOUR_DIFFS > TAU)
        HOUR_DIFFS[mask] = np.nan

        history = np.nansum(
            cme_history_function_vectorized(HOUR_DIFFS, TAU, EPSILON), axis=1
        )

        # NOTE: This has no cap
        return history

    def get_smooth_label(self, HOUR_DIFFS):
        """
        Gets a smooth probabilistic label for the row.

        The label is based on the idea of binary forecasting (so range 0 to 1), but we
        implement a smooth function that somewhat slowly rises over time from 0 to 1,
        reaching 1 at the event time and then more rapidly (but smoothly) dropping back to
        0. This is based on the combination of two gaussian functions.
        """
        H = self.smooth_label_params.H
        THETA = self.smooth_label_params.THETA
        D = self.smooth_label_params.D
        EPSILON = self.smooth_label_params.EPSILON

        label = np.nansum(
            smooth_label_function_vectorized(HOUR_DIFFS, H, THETA, D, EPSILON), axis=1
        )
        return np.minimum(label, 1)

    def get_label(self, row):
        return row["smooth_label"]

    def build_metadata_dict(self, row, label, extra_metadata: dict = {}):
        item_metadata = dict()

        # Here the binary label is based on the threshold THETA
        # NOTE: imbalance ratio is based on label being > 0, not label being > THETA
        binary_label = row["smooth_label"] > self.smooth_label_params.THETA

        item_metadata["cme_history"] = tt(row["cme_history"])
        item_metadata["cme_diff"] = tt(row["cme_diff"])
        item_metadata["verification_level"] = tt(row["verification_level"])
        item_metadata["binary_label"] = tt(binary_label)
        item_metadata["sampling_weight"] = tt(row["sampling_weight"])
        item_metadata["loss_weight"] = tt(row["loss_weight"])
        item_metadata["harpnum"] = row["harpnum"]
        item_metadata["timestamp"] = row["end_image"]

        # Add the extra metadata if any
        item_metadata.update(extra_metadata)

        return item_metadata

    def __len__(self):
        return len(self.dataset)

    def get_harpnum_iterator(self, harpnum: int):
        # Select from dataset, ordered by timestamp
        # all indices where harpnum is equal to given

        # Get the indices
        indices = (
            self.dataset[self.dataset["harpnum"] == harpnum]
            .sort_values(by="end_image", ascending=True)
            .index.tolist()
        )

        if len(indices) == 0:
            raise ValueError(f"No indices found for harpnum {harpnum}")

        # The first yield is the number of indices
        yield len(indices)

        # Next yields are the actual data
        for idx in indices:
            # Check right row
            row = self.dataset.iloc[idx].copy()
            if row["harpnum"] != harpnum:
                raise ValueError(
                    f"Row harpnum {row['harpnum']} does not match requested harpnum {harpnum}"
                )
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        return

    @abstractmethod
    def __repr__(self) -> str:
        return
