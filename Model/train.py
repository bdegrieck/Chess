import logging
import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Boards.dataloader import ChessBoardDataset
from Model.constants import ModelParams
from Model.model import ConvNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Preprocessing.helpers import load_json_dir_parallel

# Logging setup
logging.basicConfig(
    filename="training.log",  # Log file
    filemode="w",  # Overwrite the log file each time
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Log level
)


def train(
        labeled_tensors_dir: str,
        num_batches: int,
        device: str,
        num_epochs: int = 10,
        model_save_path: str = "final_model",  # Default save name
        load_model_path: str = None
) -> None:
    """
    Train a CNN model on chess board states for binary classification.

    Args:
        labeled_tensors_dir (str): Path to directory containing labeled tensors.
        num_batches (int): Batch size for training.
        device (str): Device to run training on ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs. Default is 10.
        model_save_path (str): Path to save the trained model. Default is 'final_model'.
        load_model_path (str): Path to load a pre-trained model. Default is None.

    Returns:
        None
    """
    print("Fetching data...")
    data = load_json_dir_parallel(dir=labeled_tensors_dir)
    if not data:
        raise ValueError("No data found. Ensure 'labeled_tensors_dir' contains valid tensors.")
    print("Data fetched.")

    print("Converting data to DataLoader...")
    board_states = ChessBoardDataset(data)
    dataloader = DataLoader(board_states, batch_size=num_batches, shuffle=True)
    print("DataLoader ready.")

    params = ModelParams(
        channel_values=[12, 64, 128],
        num_classes=1,
        kernel_size=3,
        stride=1,
        padding=1,
        x_size=8
    )

    device = torch.device(device)
    print(f"Using device: {device}")

    start_time = time.time()
    model = ConvNet(params=params).to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        logging.info(f"Loaded model from {load_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = BCEWithLogitsLoss()

    # Initialize lists to store metrics
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode
        epoch_loss = 0  # Reset for each epoch

        for batch_idx, batch in enumerate(dataloader):
            inputs, label = batch
            inputs, label = inputs.to(device), label.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs.float())
            loss = criterion(outputs.flatten(), label.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:  # Log every 10 batches
                logging.info(f"Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Track epoch-level loss
        epoch_losses.append(epoch_loss / len(dataloader))
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")
        scheduler.step()

        # Save model checkpoint
        checkpoint_path = f"{model_save_path}_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved model checkpoint: {checkpoint_path}")

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds.")

    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', label='Training Loss')
    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()


def test_model(test_data_dir: str, model_path: str, batch_size: int, device: str):
    # Load test data
    print("Fetching data")
    test_data = load_json_dir_parallel(dir=test_data_dir)  # Replace with your data loading function
    print("Data Fetched")
    test_dataset = ChessBoardDataset(test_data)
    print("Converting data to dataloader")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data Converted")

    # Define model parameters
    params = ModelParams(
        channel_values=[12, 32, 64],  # Must match training parameters
        num_classes=1,
        kernel_size=3,
        stride=1,
        padding=1,
        x_size=8
    )

    # Load model
    model = ConvNet(params=params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode

    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, batch in enumerate(test_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs.float())
            predictions = torch.sigmoid(outputs).flatten()  # Sigmoid for binary classification
            predictions = (predictions > 0.5).float()  # Threshold to binary values (0 or 1)

            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
