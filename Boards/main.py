import torch

from Model.constants import ModelParams
from Model.model import ConvNet
from Preprocessing.helpers import reformat_pgn, convert_png_to_tensors, load_tensor


def store_jan_2013_games() -> None:
    """
    Stores the January 2013 games from lichess database
    """
    janurary_2013_raw_file = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Janurary 2013//2013_Janurary_raw_file.pgn"
    janurary_2013_games_dir = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Janurary 2013//formatted_pgn_files"
    tensor_dir = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Janurary 2013//tensors"

    reformat_pgn(raw_file=janurary_2013_raw_file, game_limit=100, output_dir=janurary_2013_games_dir)
    convert_png_to_tensors(input_dir=janurary_2013_games_dir, output_dir=tensor_dir)


def load_jan_2013_tensor():
    tensor_path = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Janurary 2013//tensors//game_2.pt"
    tensor = load_tensor(tensor_path)
    return tensor


if __name__ == "__main__":

    board_game = load_jan_2013_tensor()
    params = ModelParams(
        channel_values=[12, 32, 64],  # Input channels and convolution layers
        num_classes=1,  # Binary classification
        kernel_size=3,
        stride=1,
        padding=1,
        x_size=8  # Chess board size is 8x8
    )
    model = ConvNet(params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

