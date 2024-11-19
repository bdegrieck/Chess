from typing import Literal
import torch

from pydantic.v1 import BaseModel

piece_map = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,  # Black pieces
}

class GameMetaData(BaseModel):
    """
    header - meta information about a game
    moves - StringIo object of all the moves of a game
    """
    header: str
    moves: str

    class Config:
        arbitrary_types_allowed = True


class Players:
    black = "Black"
    white = "White"


class BoardState(BaseModel):
    """
    player_turn: either Black or White string
    turn_num: number of the turn of the game
    board_state: tensor of the board state
    """
    player_turn: Literal[Players.white, Players.black]
    turn_num: int
    board_state: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class Directories(BaseModel):
    """
    raw_directory: Directory for the large png file
    input_directory: Directory for splitted png files
    output_directory: Directory for the tensors
    """
    raw_directory: str
    input_directory: str
    output_directory: str
