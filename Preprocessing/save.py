import os
from Preprocessing.helpers import reformat_pgn, convert_png_to_tensors, load_tensors_from_dir, put_labels_on_boards, \
    save_labeled_games_to_json


def save_files(large_pgn_file: str, formatted_pgn_dir: str, tensor_dir: str, output_labels_dir: str,  game_limit: int) -> None:
    """
    :param large_pgn_file: large pgn file from lichess.org
    :param formatted_pgn_dir: directory location of separated pgn games
    :param tensor_dir: directory of the tensors
    :param output_labels_dir: output directory for labels
    :param game_limit: number of games you want to extract from large pgn file
    :return: None
    """
    reformat_pgn(raw_file=large_pgn_file, game_limit=game_limit, output_dir=formatted_pgn_dir)
    convert_png_to_tensors(input_dir=formatted_pgn_dir, output_dir=tensor_dir)
    games = load_tensors_from_dir(input_dir=tensor_dir)
    labels = put_labels_on_boards(games=games)
    save_labeled_games_to_json(labeled_boards=labels, output_dir=output_labels_dir)
    print("games saved")


def delete_files(directories: list[str]) -> None:
    """
    :param directories: list of directories you want to be cleared
    :return: None
    """
    for directory in directories:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                # Check if it's a file and delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # For files and symbolic links
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Removes empty subdirectories
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print(f"deleted files from directory {directory}")


def feb_2013_config(game_limit: int):
    large_pgn_file = "/Users/bende/Chess Games/February 2013/lichess_db_standard_rated_2013-02.pgn"
    formatted_pgn_dir = "/Users/bende/Chess Games/February 2013/formatted_pgn_files"
    tensors_dir = "/Users/bende/Chess Games/February 2013/tensors"
    labeled_tensors_dir = "/Users/bende/Chess Games/February 2013/labeled_tensors"
    delete_files(directories=[formatted_pgn_dir, tensors_dir, labeled_tensors_dir])
    save_files(large_pgn_file=large_pgn_file, formatted_pgn_dir=formatted_pgn_dir, tensor_dir=tensors_dir,
               output_labels_dir=labeled_tensors_dir, game_limit=game_limit)


def jan_2013_config(game_limit: int):
    large_pgn_file = "/Users/bende/Chess Games/Janurary 2013/2013_Janurary_raw_file.pgn"
    formatted_pgn_dir = "/Users/bende/Chess Games/Janurary 2013/formatted_pgn_files"
    tensors_dir = "/Users/bende/Chess Games/Janurary 2013/tensors"
    labeled_tensors_dir = "/Users/bende/Chess Games/Janurary 2013/labeled tensors"
    delete_files([formatted_pgn_dir, tensors_dir, labeled_tensors_dir])
    save_files(large_pgn_file=large_pgn_file, formatted_pgn_dir=formatted_pgn_dir, tensor_dir=tensors_dir,
               output_labels_dir=labeled_tensors_dir, game_limit=game_limit)


def test_config():
    large_pgn_file = "/Users/bende/Chess Games/Test/lichess_db_standard_rated_2013-03.pgn"
    formatted_pgn_dir = "/Users/bende/Chess Games/Test/formatted_pgn_files"
    tensors_dir = "/Users/bende/Chess Games/Test/tensors"
    labeled_tensors_dir = "/Users/bende/Chess Games/Test/labeled_tensors"
    delete_files(directories=[formatted_pgn_dir, tensors_dir, labeled_tensors_dir])
    save_files(large_pgn_file=large_pgn_file, formatted_pgn_dir=formatted_pgn_dir, tensor_dir=tensors_dir,
               output_labels_dir=labeled_tensors_dir, game_limit=100)
