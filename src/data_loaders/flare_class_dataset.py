from .base_swan_dataset import BaseSWANDataset
import numpy as np


class FlareClassSWANDataset(BaseSWANDataset):
    """Dataset for SWAN that returns whether a region
    produces a flar above the given threshold
    in the next T hours.
    Args:
        BaseSWANDataset ([type]): [description]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_pos_weight(self):
        pos_count = np.sum(self._periods[:, 4])

        neg_count = len(self._periods[:, 4]) - pos_count

        if pos_count == 0:
            return 0

        return neg_count / pos_count

    def _get_label(self, row) -> int:
        return int(row[4])

    def _get_metadata(self, row) -> dict:
        base_metadata = super()._get_base_metadata(row)

        base_metadata["flare_id"] = row[6]

        return base_metadata

    def _get_all_labels(self) -> np.ndarray:
        return self._periods[:, 4].astype(int)

    def _get_baseline_predictions(self) -> np.ndarray:
        return self._periods[:, 14].astype(int)  # is_active
