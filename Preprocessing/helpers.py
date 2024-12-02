import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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


def process_game_file(game_file: str, output_dir: str, game_num: int) -> None:
    """
    Processes a single game file and saves the resulting tensors.
    :param game_file: Path to the game file.
    :param output_dir: Directory to save tensors.
    :param game_num: Index of the game for naming output file.
    """
    with open(game_file, 'r') as file:
        content = json.load(file)

    moves = StringIO(content.get("moves"))
    game = chess.pgn.read_game(moves)
    board = game.board()

    board_states: list[torch.tensor] = []
    for move in game.mainline_moves():
        board.push(move)
        tensor = board_to_tensor(board=board)
        board_states.append(tensor)

    torch.save([state for state in board_states], f=f'{output_dir}/game_{game_num}.pt')


def convert_png_to_tensors_parallel(input_dir: str, output_dir: str, num_workers: int = 4) -> None:
    """
    Converts PGN files to tensors using parallel processing.
    :param input_dir: Directory of the PGN files.
    :param output_dir: Directory to save the tensors.
    :param num_workers: Number of parallel workers.
    :return: None
    """
    # Collect all the files from the input directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Start parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for game_num, game_file in enumerate(files, start=1):
            executor.submit(process_game_file, game_file, output_dir, game_num)


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


def save_single_board_to_json(board: BoardStateLabeled, output_dir: str, idx: int) -> None:
    """
    Saves a single labeled board to a JSON file.
    :param board: Labeled board to save.
    :param output_dir: Directory to save the JSON file.
    :param idx: Index of the board for file naming.
    """
    labeled_board_dict = {
        MetaDataKeys.board_state: board.board_state.tolist(),
        MetaDataKeys.label: board.label
    }

    output_file = os.path.join(output_dir, f"board_{idx}.json")
    with open(output_file, "w") as f:
        json.dump(labeled_board_dict, f)


def save_labeled_games_to_json_parallel(labeled_boards: list[BoardStateLabeled], output_dir: str, num_workers: int = 4) -> None:
    """
    Saves labeled boards to JSON files in parallel.
    :param labeled_boards: List of labeled boards.
    :param output_dir: Directory to save labeled data.
    :param num_workers: Number of parallel workers.
    :return: None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Start parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(save_single_board_to_json, board, output_dir, idx)
            for idx, board in enumerate(labeled_boards)
        ]

        # Optionally wait for all futures to complete (useful for error handling/logging)
        for future in futures:
            future.result()  # Raise exceptions, if an


# def load_tensor(file_path: str) -> list[dict[str, Any]]:
#     """
#     :param file_path: file path to load tensor from which is a whole game of a list of board states
#     """
#     loaded_tensor = torch.load(file_path)
#     return loaded_tensor
#
#
# def load_tensors_from_dir(input_dir: str) -> list[list[dict[str, Any]]]:
#     files = [os.path.join(input_dir + f"/{f}") for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
#     games = []
#     for file in files:
#         games.append(load_tensor(file_path=file))
#     return games


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
    flipped_positions = set()
    for _ in range(num_flips):
        channel = random.randint(0, 11)
        row = random.randint(0, 7)
        column = random.randint(0, 7)
        if (channel, row, column) not in flipped_positions:
            flipped_positions.add((channel, row, column))
            corrupted_board[channel, row, column] = 1 - corrupted_board[channel, row, column]
        return corrupted_board


# def put_labels_on_boards(games: list[list[torch.tensor]]) -> list[BoardStateLabeled]:
#     """
#     :param games: List of board games
#     :return: list of labeled board games
#     """
#     labeled_data = []
#     for game in games:
#         label_key = create_split_dict(max_key=len(game))
#
#         for idx, board_state in enumerate(game, start=1):
#             if label_key.get(idx) == 1:
#                 board_state = corrupt_board(board_state=board_state, num_flips=20)
#
#             labeled_data.append(BoardStateLabeled(
#                     board_state=board_state,
#                     label=label_key.get(idx)
#                 )
#             )
#
#     return labeled_data


def process_games(input_dir: str) -> list[BoardStateLabeled]:
    """
    Load tensors from a directory and label the board states in a single function.

    :param input_dir: Directory containing tensor files.
    :return: List of labeled board states.
    """
    labeled_data = []
    files = [os.path.join(input_dir + f"/{f}") for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file in files:
        game = torch.load(file)
        label_key = create_split_dict(max_key=len(game))

        for idx, board_state in enumerate(game, start=1):
            if label_key.get(idx) == 1:
                board_state = corrupt_board(board_state=board_state, num_flips=20)

            labeled_data.append(BoardStateLabeled(
                board_state=board_state,
                label=label_key.get(idx)
            ))

    return labeled_data


def load_file(file_path):
    if ".DS_Store" in file_path:
        return None
    with open(file_path, "r") as f:
        data_dict = json.load(f)
        data_dict["board_state"] = torch.tensor(data_dict["board_state"])
        return data_dict


def load_json_dir_parallel(dir: str) -> list:
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    data = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(load_file, files)
        data = [result for result in results if result is not None]
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
