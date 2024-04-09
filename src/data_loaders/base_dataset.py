from torch.utils.data import Dataset
import os


class BaseDataset(Dataset):
    def __init__(self):
        """
        Initialize the dataset.
        This method should be overridden by subclass.
        """
        super(BaseDataset, self).__init__()

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
        - Integer count of samples.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
        - idx (int): The index of the sample to fetch.

        Returns:
        - A sample fetched using the index.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def get_dataloader_weights(self):
        """
        Returns the weights of the dataset for each sample.
        """
        raise NotImplementedError("This method should be overridden by subclass")
