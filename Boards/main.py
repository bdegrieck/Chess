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
    print(tensor)


if __name__ == "__main__":
    load_jan_2013_tensor()