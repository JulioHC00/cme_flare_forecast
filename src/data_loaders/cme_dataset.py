from .base_swan_dataset import BaseSWANDataset
import pandas as pd
import numpy as np


class CMESWANDataset(BaseSWANDataset):
    """Dataset for SWAN that returns whether a region
    produces a flar above the given threshold
    in the next T hours.
    Args:
        BaseSWANDataset ([type]): [description]
    """

    def __init__(self, no_flare_cmes=True, **kwargs):
        # no_flare_cmes is to decide whether we want to
        # include CMEs that don't have an associated flare

        super().__init__(**kwargs)

        self.no_flare_cmes = no_flare_cmes

    def _process_periods(self, periods) -> pd.DataFrame:
        if not self.no_flare_cmes:
            # We need to remove all rows where there's
            # a CME with no flare and none with an associated flare
            condition = ~(  # This negates the condition
                (periods["has_cme_flare_above_threshold"] == 0)
                & (  # If there's no flare above threshold with CME
                    periods["has_cme_flare_above_threshold"] == 0
                )
                & (  # Nor below threshold
                    periods["has_cme_no_flare"] == 1
                )  # But there's a CME with no flare
            )  # Drop it

            periods = periods.loc[condition].copy()

        self.has_cme_idx = len(periods.columns)

        # Either way we need to create a new combind column for the labels
        periods["has_cme"] = (
            periods["has_cme_flare_above_threshold"]
            | periods["has_cme_flare_below_threshold"]
            | periods["has_cme_no_flare"]
        ).astype(int)

        # And a CME_type column that's 0 if no CME, 1 if CME with flare above threshold
        # 2 if CME with flare below threshold and 3 if CME with no flare
        # Keep in mind a row may have multiple at the same time so we need to
        # choose the highest priority one

        def get_cme_type(row):
            if row["has_cme"] == 0:
                return 0
            elif row["has_cme_flare_above_threshold"] == 1:
                return 1
            elif row["has_cme_flare_below_threshold"] == 1:
                return 2
            elif row["has_cme_no_flare"] == 1:
                return 3

        def get_flare_id(row):
            if row["has_cme"] == 0:
                return None
            elif row["has_cme_flare_above_threshold"] == 1:
                return row["cme_flare_above_threshold_id"]
            elif row["has_cme_flare_below_threshold"] == 1:
                return row["cme_flare_below_threshold_id"]
            elif row["has_cme_no_flare"] == 1:
                return None

        def get_cme_id(row):
            if row["has_cme"] == 0:
                return None
            elif row["has_cme_flare_above_threshold"] == 1:
                return row["cme_flare_above_threshold_id"]
            elif row["has_cme_flare_below_threshold"] == 1:
                return row["cme_flare_below_threshold_id"]
            elif row["has_cme_no_flare"] == 1:
                return row["cme_no_flare_id"]

        # Get the current number of columns
        n_columns = len(periods.columns)

        self.cme_type_idx = n_columns
        self.cme_flare_id_idx = n_columns + 1
        self.cme_id_idx = n_columns + 2

        periods["cme_type"] = periods.apply(get_cme_type, axis=1)
        periods["cme_flare_id"] = periods.apply(get_flare_id, axis=1)
        periods["cme_id"] = periods.apply(get_cme_id, axis=1)

        return periods

    def _get_pos_weight(self):
        pos_count = np.sum(self._periods[:, self.has_cme_idx])

        neg_count = len(self._periods) - pos_count

        if pos_count == 0:
            return 0

        return neg_count / pos_count

    def _get_label(self, row) -> int:
        return int(row[self.has_cme_idx])

    def _get_metadata(self, row) -> dict:
        base_metadata = super()._get_base_metadata(row)

        base_metadata["flare_id"] = row[self.cme_flare_id_idx]
        base_metadata["cme_id"] = row[self.cme_id_idx]
        base_metadata["cme_type"] = row[self.cme_type_idx]

        return base_metadata

    def _get_all_labels(self) -> np.ndarray:
        return self._periods[:, self.has_cme_idx].astype(int)

    def _get_baseline_predictions(self) -> np.ndarray:
        return self._periods[:, 14].astype(int)  # is_active
