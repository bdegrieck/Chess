import time
from Model.train import train, test_model
from Model.tuning import hyperparameter_tuning
from Preprocessing.helpers import tensor_to_fen, load_file, process_games, \
    save_labeled_games_to_json
from Preprocessing.save import save_files, delete_files


class Devices:
    cuda = "cuda"
    cpu = "cpu"


def trainer():
    dir = "C://Users//bende//Chess Games//Janurary 2013//labeled_tensors"
    model_destination_path = "C://Users//bende//Chess//Model//Saved Models//cudatime.pth"
    train(labeled_tensors_dir=dir, num_batches=64, device=Devices.cpu, num_epochs=5,
          model_save_path=model_destination_path)


def model_performance():
    labeled_tensors = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Test//labeled_tensors"
    model_path = "C://Users//bende//Chess//Model//Saved Models//jan2023.pth"
    test_model(test_data_dir=labeled_tensors, batch_size=64, model_path=model_path, device=Devices.cuda)


def convert_file_fen(file_path: str):
    json = load_file(file_path)
    fen_str = tensor_to_fen(board_state=json["board_state"])
    print(fen_str)


def hyperparam_tune():
    dir = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Janurary 2013//labeled tensors"
    model_save_path = "C://Users//bende//Chess//Model//Saved Models//model1.pth"
    hyperparameter_tuning(labeled_tensors_dir=dir, device=Devices.cuda, model_save_path=model_save_path)


if __name__ == "__main__":
    output_directory = "C://Users//bende//Chess Games//Janurary 2013//labeled_tensors"
    tensor_dir = "C://Users//bende//Chess Games//Janurary 2013//tensors"
    print(f"getting labels")
    labels = process_games(input_dir=tensor_dir)
    print(f"saving labels")
    save_labeled_games_to_json(labeled_boards=labels, output_dir=output_directory)
