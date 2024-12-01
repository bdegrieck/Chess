from typing import Any
from torch.utils.data import Dataset, DataLoader

from Preprocessing.constants import MetaDataLabels, MetaDataKeys


class ChessBoardDataset(Dataset):

    def __init__(self, data: list[dict[str, Any]]):
        """
        Initialize the dataset with board states and labels.
        :param data: List of dictionaries containing board states and labels
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve a single item from the dataset.
        """
        board_state = self.data[idx][MetaDataKeys.board_state]
        label = self.data[idx][MetaDataKeys.label]
        return board_state, label
