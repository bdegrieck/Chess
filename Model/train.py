import logging

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from Boards.dataloader import ChessBoardDataset
from Model.constants import ModelParams
from Model.model import ConvNet
from Preprocessing.helpers import load_json_dir
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(
    labeled_tensors_dir: str,
    num_batches: int,
    num_epochs: int = 10,
    model_save_path: str = "final_model",  # Default save name
    load_model_path: str = None
):
    data = load_json_dir(dir=labeled_tensors_dir)
    board_states = ChessBoardDataset(data)
    dataloader = DataLoader(board_states, batch_size=num_batches, shuffle=True)

    params = ModelParams(
        channel_values=[12, 32, 64],  # Input channels and convolution layers
        num_classes=1,  # Binary classification
        kernel_size=3,
        stride=1,
        padding=1,
        x_size=8  # Chess board size is 8x8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvNet(params=params).to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        logging.info(f"Loaded model from {load_model_path}")
        print(f"Loaded model {load_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0  # Track total loss for the epoch
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            inputs, label = batch
            inputs, label = inputs.to(device), label.to(device)

            # Forward pass
            outputs = model(inputs.float())
            loss = criterion(outputs.flatten(), label.float())  # Compute loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate and display loss
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:  # Print every 10 batches
                logging.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

                # Display predictions and labels for debugging
                predictions = torch.sigmoid(outputs.flatten()).detach().cpu().numpy()  # Apply sigmoid for probabilities
                labels = label.cpu().numpy()
                logging.info(f"Predictions: {predictions}, Labels: {labels}")

        # Epoch-level loss summary
        logging.info(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(dataloader):.4f}")

        # Update learning rate
        scheduler.step()

    # Save the model only after the last epoch
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model saved at: {model_save_path}.pt")
    print(f"Training completed. Model saved at {model_save_path}")


def test_model(test_data_dir: str, model_path: str, batch_size: int):
    # Load test data
    test_data = load_json_dir(dir=test_data_dir)  # Replace with your data loading function
    test_dataset = ChessBoardDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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