from io import StringIO
import chess.pgn
import chess
import torch
import json
import os
import random
from typing import Any

from Preprocessing.constants import PIECE_MAP, GameMetaData, Players, BoardState, MetaDataKeys, BoardStateLabeled, \
    MetaDataLabels, PIECE_MAP_REVERSED


def reformat_pgn(raw_file: str, game_limit: int, output_dir: str) -> None:
    """
    :param output_dir: output directory
    :param raw_file: File of the large png file
    :param game_limit: Amount of games you want to extract
    :return: None
    """

    with open(raw_file, 'r') as file:
        content = file.read()

    # Split games based on "[Event" which starts a new game entry
    splitter = content.split("[Event ")

    # if user wants all the games
    if game_limit == -1:
        games = [game for game in splitter if game.strip()]
    else:
        games = [game for game in splitter if game.strip()][:game_limit + 1]

    # extract game data of its headers and moves
    meta_data_list = []
    for game in games:
        game_stripped = game.strip().split("\n\n", 1)
        meta_data = GameMetaData(
            header=game_stripped[0],
            moves=game_stripped[1]
        )
        meta_data_list.append(meta_data)

    # save individual pgn files
    save_pgn_to_json(meta_data_list=meta_data_list, output_dir=output_dir)


def save_pgn_to_json(meta_data_list: list[GameMetaData], output_dir: str) -> None:
    """
    :param meta_data_list: list of metadata of games
    :param output_dir: directory you want to put json files in
    :return: None
    """

    for idx, game in enumerate(meta_data_list, start=1):
        game_dict = game.dict()
        output_file = os.path.join(output_dir, f"game_{idx}.pgn")

        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(game_dict, f, indent=4)


def convert_png_to_tensors(input_dir: str, output_dir: str) -> None:
    """
    :param input_dir: directory of the pgn files
    :param output_dir: directory you want to save the tensors to
    :return: None
    """

    # collects all the files from the input directory
    files = [os.path.join(input_dir + f"/{f}") for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # iterate trough each file. We start at one since the first one is .DS_STORE
    for game_num, game_file in enumerate(files[1:], start=1):
        with open(game_file, 'r') as file:
            content = json.load(file)

        moves = StringIO(content.get("moves"))
        game = chess.pgn.read_game(moves)
        board = game.board()

        board_states: list[BoardState] = []
        move_number = 1
        for move in game.mainline_moves():
            board.push(move)
            tensor = board_to_tensor(board=board)
            player = Players.black if board.turn else Players.white
            board_states.append(BoardState(
                player_turn=player,
                turn_num=move_number,
                board_state=tensor
                )
            )

            # Only increment move number after Black's move, as one full move includes both White and Black
            if board.turn:
                move_number += 1

        torch.save([state.dict() for state in board_states], f=f'{output_dir}/game_{game_num}.pt')


def board_to_tensor(board):
    """
    :param board: pychess board object that is converted to tensors for each move
    :return: board_tensor: tensor of the board matrix
    """
    board_tensor = torch.zeros(12, 8, 8, dtype=torch.float)  # 8x8 tensor for chess board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            channel = PIECE_MAP[piece.symbol()]
            board_tensor[channel, row, col] = 1
    return board_tensor


def save_labeled_games_to_json(games: list[MetaDataLabels], output_dir: str) -> None:
    """
    :param games: list of labeled games
    :param output_dir: directory you want to store labeled data
    :return: None
    """
    for idx, game in enumerate(games, start=1):

        game_dict = {
            MetaDataKeys.board_state: [{
                MetaDataKeys.board_state: board.board_state.tolist(),
                MetaDataKeys.label: board.label,
                MetaDataKeys.player_turn: board.player_turn,
                MetaDataKeys.turn_num: board.turn_num
            } for board in game.board_states],
            MetaDataKeys.corrupted_labels: game.corrupted_labels,
            MetaDataKeys.non_corrupted_labels: game.non_corrupted_labels
        }

        output_file = os.path.join(output_dir, f"game_{idx}.json")
        with open(output_file, "w") as f:
            json.dump(game_dict, f)



def load_tensor(file_path: str) -> list[dict[str, Any]]:
    """
    :param file_path: file path to load tensor from which is a whole game of a list of board states
    """
    loaded_tensor = torch.load(file_path)
    return loaded_tensor


def load_tensors_from_dir(input_dir: str) -> list[list[dict[str, Any]]]:
    files = [os.path.join(input_dir + f"/{f}") for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    games = []
    for file in files:
        games.append(load_tensor(file_path=file))
    return games


def create_split_dict(max_key: int):
    """
    Create a dictionary where keys are integers from 1 to max_key,
    and values are split 50/50 between 0 and 1.

    :param max_key: The maximum key value in the dictionary.
    :return: A dictionary with keys as integers and values split 50/50 between 0 and 1.
    """
    keys = list(range(1, max_key + 1))

    # Split keys randomly into two equal groups
    random.shuffle(keys)
    mid_point = len(keys) // 2
    values = [0] * mid_point + [1] * (len(keys) - mid_point)
    random.shuffle(values)

    return dict(zip(keys, values))


def corrupt_board(board_state, num_flips: int = 1):
    """
    Corrupt the board state by flipping a random bit.
    :param board_state: Tensor of shape (12, 8, 8) representing the board state.
    :param num_flips: Number of bits to flip
    :return: Corrupted board state.
    """
    corrupted_board = board_state.clone()
    for _ in range(num_flips):
        channel = random.randint(0, 11),  # Random channel
        row = random.randint(0, 7),  # Random row
        column = random.randint(0, 7),  # Random column
        corrupted_board[channel, row, column] = 1 - corrupted_board[channel, row, column]  # Flip the bit (0 -> 1, 1 -> 0)
        return corrupted_board


def put_labels_on_boards(games: list[list[dict[str, Any]]]) -> list[MetaDataLabels]:
    """
    :param games: List of board games
    :return: list of labeled board games
    """
    data = []
    for game in games:
        labeled_data = []
        label_key = create_split_dict(max_key=len(game))

        for idx, board_state in enumerate(game, start=1):
            if label_key.get(idx) == 1:
                board_state[MetaDataKeys.board_state] = corrupt_board(board_state=board_state.get(MetaDataKeys.board_state), num_flips=5)

            labeled_data.append(BoardStateLabeled(
                    player_turn=board_state.get(MetaDataKeys.player_turn),
                    turn_num=board_state.get(MetaDataKeys.turn_num),
                    board_state=board_state.get(MetaDataKeys.board_state),
                    label=label_key.get(idx)
                )
            )

        data.append(MetaDataLabels(
            non_corrupted_labels=[idx for idx, val in label_key.items() if val == 0],
            corrupted_labels=[idx for idx, val in label_key.items() if val == 1],
            board_states=labeled_data
            )
        )

    return data

def load_json_dir(dir: str) -> list:
    """
    :param dir: directory file path
    :return: list of data
    """
    files = [os.path.join(dir + f"/{f}") for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    data = []
    for file in files:
        with open(file, "r") as f:
            data_dict = json.load(f)

            for board in data_dict[MetaDataKeys.board_state]:
                board[MetaDataKeys.board_state] = torch.tensor(board[MetaDataKeys.board_state])
            data.append(data_dict)
    return data


def tensor_to_fen(board_state: torch.tensor) -> str:
    """
    :param board_state: tensor of board state
    :return: FEN string of board state
    """
    fen_rows = []
    for row in range(8):
        fen_row = []
        empty_count = 0

        for col in range(8):
            piece_found = False

            for channel in range(12):
                if board_state[channel, row, col] == 1:
                    fen_row.append(PIECE_MAP_REVERSED[channel])
                    piece_found = True
                    break

            if not piece_found:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row.append(str(empty_count))
                    empty_count = 0

        if empty_count > 0:
            fen_row.append(str(empty_count))
        fen_rows.append(''.join(fen_row))

    fen_board = '/'.join(fen_rows)
    fen_string = f"{fen_board} w KQkq - 0 1"

    return fen_string
