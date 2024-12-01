from typing import Any
from torch.utils.data import Dataset, DataLoader

from Preprocessing.constants import MetaDataLabels, MetaDataKeys


class ChessBoardDataset(Dataset):

    def __init__(self, data: list[dict[str, Any]], return_extra: bool = False):
        """
        Initialize the dataset with board states and labels.
        :param data: List of dictionaries containing board states and labels
        :param return_extra: boolean to return extra attributes in getitem
        """
        self.data = data
        self.return_extra = return_extra

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve a single item from the dataset.
        """
        board_state = self.data[idx][MetaDataKeys.board_state][idx][MetaDataKeys.board_state]
        label = self.data[idx][MetaDataKeys.board_state][idx][MetaDataKeys.label]
        if self.return_extra:
            player_turn = self.data[idx]['player_turn']
            turn_num = self.data[idx]['turn_num']
            return board_state, label, player_turn, turn_num
        return board_state, label
