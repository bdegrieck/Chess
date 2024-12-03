import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from itertools import product
import matplotlib.pyplot as plt

from Boards.dataloader import ChessBoardDataset
from Model.constants import ModelParams
from Model.model import ConvNet
from Preprocessing.helpers import load_json_dir_parallel


def train_with_params(
    labeled_tensors_dir: str,
    num_batches: int,
    device: str,
    num_epochs: int,
    lr: float,
    model_params: ModelParams,
    model_save_path: str,
) -> float:
    """
    Train a CNN model with specific hyperparameters.

    Args:
        labeled_tensors_dir (str): Path to directory containing labeled tensors.
        num_batches (int): Batch size for training.
        device (str): Device to run training on ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        model_params (ModelParams): Model parameters.
        model_save_path (str): Path to save the trained model.

    Returns:
        float: Final training loss.
    """
    # Load data
    data = load_json_dir_parallel(dir=labeled_tensors_dir)
    if not data:
        raise ValueError("No data found. Ensure 'labeled_tensors_dir' contains valid tensors.")
    board_states = ChessBoardDataset(data)
    dataloader = DataLoader(board_states, batch_size=num_batches, shuffle=True)

    # Initialize model, optimizer, scheduler, and loss function
    device = torch.device(device)
    model = ConvNet(params=model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = BCEWithLogitsLoss()

    # Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            inputs, label = batch
            inputs, label = inputs.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs.flatten(), label.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(dataloader))
        scheduler.step()

    # Return the final loss
    return epoch_losses[-1]


def hyperparameter_tuning(
    labeled_tensors_dir: str,
    device: str,
    model_save_path: str = "final_model",
):
    """
    Perform hyperparameter tuning for the CNN model.

    Args:
        labeled_tensors_dir (str): Path to directory containing labeled tensors.
        device (str): Device to run training on ('cuda' or 'cpu').
        model_save_path (str): Path to save the trained model.

    Returns:
        list: Hyperparameter tuning results including combinations and final losses.
    """
    # Define hyperparameter grid
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [16, 32]
    channel_configs = [[12, 32], [12, 64]]
    num_epochs = 10

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(product(learning_rates, batch_sizes, channel_configs))

    # Collect results for plotting
    results = []
    for lr, batch_size, channels in hyperparameter_combinations:
        print(f"Training with: lr={lr}, batch_size={batch_size}, channels={channels}")

        # Create model params
        params = ModelParams(
            channel_values=channels,
            num_classes=1,
            kernel_size=3,
            stride=1,
            padding=1,
            x_size=8
        )

        # Train the model and get the final loss
        final_loss = train_with_params(
            labeled_tensors_dir=labeled_tensors_dir,
            num_batches=batch_size,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            model_params=params,
            model_save_path=model_save_path,
        )

        # Append results
        results.append({
            "lr": lr,
            "batch_size": batch_size,
            "channels": channels,
            "final_loss": final_loss,
        })

    return results


def plot_hyperparameter_tuning_results(results):
    """
    Plot training error for each hyperparameter combination.

    Args:
        results (list of dict): Each dict contains 'lr', 'batch_size', 'channels', and 'final_loss'.
    """
    # Extract data for plotting
    labels = [
        f"lr={res['lr']}, bs={res['batch_size']}, ch={res['channels']}"
        for res in results
    ]
    losses = [res['final_loss'] for res in results]

    # Calculate min and max loss for adjusting the x-axis scale
    min_loss = min(losses)
    max_loss = max(losses)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.barh(labels, losses, color='skyblue')
    plt.xlabel("Final Training Loss", fontsize=14)
    plt.ylabel("Hyperparameter Combinations", fontsize=14)
    plt.title("Training Loss for Different Hyperparameter Combinations", fontsize=16)

    # Set a narrower x-axis range with some padding
    plt.xlim(min_loss - 0.01, max_loss + 0.01)  # Adjust padding as needed
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

