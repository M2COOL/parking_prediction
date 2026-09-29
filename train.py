import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import ParkingDataset
from models import create_model
from utils.metrics import mae, rmse

from utils.plot import plot_loss, plot_prediction


# ============================================================
# 配置
# ============================================================

MODEL_NAME = "lstm"

NUM_SAMPLES = 2000
SEQ_LEN = 12

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

HIDDEN_SIZE = 64
NUM_LAYERS = 2

TRAIN_RATIO = 0.8

MODEL_SAVE_PATH = f"checkpoints/{MODEL_NAME}.pth"


def main():

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Sequence    : {SEQ_LEN}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print("=" * 60)


    # ========================================================
    # Dataset
    # ========================================================

    dataset = ParkingDataset(
        num_samples=NUM_SAMPLES,
        seq_len=SEQ_LEN
    )

    train_size = int(
        len(dataset) * TRAIN_RATIO
    )

    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ========================================================
    # Model
    # ========================================================

    model = create_model(
        MODEL_NAME,
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    )

    model = model.to(device)

    print(model)


    # ========================================================
    # Loss & Optimizer
    # ========================================================

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # Train
    # ========================================================

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            # Forward
            pred = model(X)

            # Loss
            loss = criterion(pred, y)

            # Backward
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()



        train_loss /= len(train_loader)
        train_losses.append(train_loss)


        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_loss = 0.0

        all_pred = []
        all_y = []

        with torch.no_grad():

            for X, y in test_loader:

                X = X.to(device)
                y = y.to(device)

                pred = model(X)

                loss = criterion(pred, y)

                val_loss += loss.item()

                all_pred.append(pred)
                all_y.append(y)

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        all_pred = torch.cat(all_pred)
        all_y = torch.cat(all_y)

        val_mae = mae(
            all_pred,
            all_y
        )

        val_rmse = rmse(
            all_pred,
            all_y
        )


        # ====================================================
        # Log
        # ====================================================

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"MAE: {val_mae:.6f} "
            f"RMSE: {val_rmse:.6f}"
        )


    # ========================================================
    # Save Model
    # ========================================================

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

    # ========================================================
    # Plot
    # ========================================================

    plot_loss(
        train_losses,
        val_losses,
        f"results/{MODEL_NAME}_loss.png"
    )

    print()
    print(f"Model saved to: {MODEL_SAVE_PATH}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    main()