from Preprocessing.helpers import reformat_pgn, convert_png_to_tensors, load_tensors_from_dir, put_labels_on_boards, \
    save_labeled_games


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
    save_labeled_games(games=labels, output_dir=output_labels_dir)
    print("games saved")
