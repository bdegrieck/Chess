from typing import Any
from torch.utils.data import Dataset, DataLoader


class ChessBoardDataset(Dataset):
    def __init__(self, board_game: list[dict[str, Any]], return_extra: bool):
        """
        Initialize the dataset with board states and labels.
        :param board_game: List of dictionaries containing board states and labels
        :param return_extra: boolean to return extra attributes in getitem
        """
        self.data = board_game
        self.return_extra = return_extra

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset.
        """
        board_state = self.data[idx]['board_state']  # 12x8x8 tensor
        label = self.data[idx].get('label', 0)  # 0 for valid, 1 for corrupted
        if self.return_extra:
            player_turn = self.data[idx]['player_turn']
            turn_num = self.data[idx]['turn_num']
            return board_state, label, player_turn, turn_num
        return board_state, label


# Example: Creating the dataset
dataset = ChessBoardDataset(board_game)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)



