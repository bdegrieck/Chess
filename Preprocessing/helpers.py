from io import StringIO
import chess.pgn
import chess
import torch
import json
import os

from Preprocessing.constants import piece_map, GameMetaData, Players, BoardState


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
        games = [game for game in splitter if game.strip()][:game_limit - 1]

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

    # iterate trough each file. We start at one stince the first one is .DS_STORE
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
    :return: board_tensor: tensor of the baord matrix
    """
    board_tensor = torch.zeros(8, 8, dtype=torch.int8)  # 8x8 tensor for chess board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            board_tensor[row, col] = piece_map[piece.symbol()]
    return board_tensor
