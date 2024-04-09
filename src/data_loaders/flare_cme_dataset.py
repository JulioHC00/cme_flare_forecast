from .base_swan_dataset import BaseSWANDataset
import numpy as np
import pandas as pd


class FlareCMESWANDataset(BaseSWANDataset):
    """Dataset for SWAN that returns whether a region
    produces a flar above the given threshold
    in the next T hours.
    Args:
        BaseSWANDataset ([type]): [description]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_pos_weight(self):
        pos_count = np.sum(self._periods[:, 9])

        neg_count = len(self._periods[:, 9]) - pos_count

        return neg_count / pos_count

    def _get_all_labels(self) -> np.ndarray:
        return self._periods[:, 9].astype(int)

    def _process_periods(self, periods: pd.DataFrame) -> pd.DataFrame:
        # For this type of dataset, we need to choose
        # only rows where there's a flare above the threshold

        # Need to translate line above to use numpy
        processed_periods = periods[periods["has_flare_above_threshold"] == 1]

        return processed_periods

    def _get_label(self, row) -> int:
        return int(row[9])

    def _get_metadata(self, row) -> dict:
        base_metadata = super()._get_base_metadata(row)

        base_metadata["flare_id"] = row[6]
        base_metadata["cme_id"] = row[11]

        return base_metadata

    def _get_baseline_predictions(self) -> np.ndarray:
        return self._periods[:, 15].astype(int)  # is_cme_active
