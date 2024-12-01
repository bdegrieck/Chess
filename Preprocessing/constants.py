from dataclasses import dataclass
from typing import Literal, Any
import torch

from pydantic.v1 import BaseModel

PIECE_MAP = {
    'P': 0,   # White Pawn
    'N': 1,   # White Knight
    'B': 2,   # White Bishop
    'R': 3,   # White Rook
    'Q': 4,   # White Queen
    'K': 5,   # White King
    'p': 6,   # Black Pawn
    'n': 7,   # Black Knight
    'b': 8,   # Black Bishop
    'r': 9,   # Black Rook
    'q': 10,  # Black Queen
    'k': 11   # Black King
}

PIECE_MAP_REVERSED = {
    0: 'P',   # White Pawn
    1: 'N',   # White Knight
    2: 'B',   # White Bishop
    3: 'R',   # White Rook
    4: 'Q',   # White Queen
    5: 'K',   # White King
    6: 'p',   # Black Pawn
    7: 'n',   # Black Knight
    8: 'b',   # Black Bishop
    9: 'r',   # Black Rook
    10: 'q',  # Black Queen
    11: 'k'   # Black King
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


class BoardStateLabeled(BaseModel):
    """
    player_turn: either Black or White string
    turn_num: number of the turn of the game
    board_state: tensor of the board state
    label: 0 if the board is not corrupted, 1 if the board is corrupted
    """
    player_turn: Literal[Players.white, Players.black]
    turn_num: int
    board_state: torch.Tensor
    label: int

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


class MetaDataKeys:
    player_turn = "player_turn"
    turn_num = "turn_num"
    board_state = "board_state"
    board_states = "board_states"
    corrupted_labels = "corrupted_labels"
    non_corrupted_labels = "non_corrupted_labels"
    label = "label"


@dataclass
class MetaDataLabels:
    non_corrupted_labels: list[int]
    corrupted_labels: list[int]
    board_states: list[BoardStateLabeled]
