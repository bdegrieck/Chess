from Model.constants import ModelParams
from Model.train import train, test_model
from Model.tuning import hyperparameter_tuning, plot_hyperparameter_tuning_results
from Preprocessing.helpers import tensor_to_fen, load_file


class Devices:
    cuda = "cuda"
    cpu = "cpu"


# Common functions I used in main so I don't have to retype stuff

def trainer(device: str):
    dir = "C://Users//bende//Chess Games//Janurary 2013//labeled_tensors"
    model_destination_path = "C://Users//bende//Chess//Model//Saved Models//cudatime.pth"

    params = ModelParams(
        channel_values=[12, 64, 128],
        num_classes=1,
        kernel_size=3,
        stride=1,
        padding=1,
        x_size=8
    )

    train(labeled_tensors_dir=dir, batch_size=16, device=device, num_epochs=10,
          model_save_path=model_destination_path, learning_rate=0.001, params=params)


def model_performance():
    params = ModelParams(
        channel_values=[12, 64, 128],
        num_classes=1,
        kernel_size=3,
        stride=1,
        padding=1,
        x_size=8
    )
    labeled_tensors = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Test//labeled_tensors"
    model_path = "C://Users//bende//Chess//Model//Saved Models//cudatime.pth_epoch_8.pth"
    test_model(test_data_dir=labeled_tensors, batch_size=64, model_path=model_path, device=Devices.cuda, params=params)


def convert_file_fen(file_path: str):
    json = load_file(file_path)
    fen_str = tensor_to_fen(board_state=json["board_state"])
    print(fen_str)


def hyperparam_tune():
    dir = "C://Users//bende//OneDrive//OU//Parallel Computing//Chess Games//Janurary 2013//labeled tensors"
    model_save_path = "C://Users//bende//Chess//Model//Saved Models//model1.pth"
    results = hyperparameter_tuning(labeled_tensors_dir=dir, device=Devices.cuda, model_save_path=model_save_path)
    plot_hyperparameter_tuning_results(results=results)


if __name__ == "__main__":
    trainer(device=Devices.cuda)
