import os

import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, save_path):
    """
    绘制训练集和验证集 Loss 曲线
    """

    plt.figure(figsize=(8, 5))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.show()

    plt.close()


def plot_prediction(y_true, y_pred, save_path, num_points=200):
    """
    绘制真实值和预测值对比图

    num_points:
        只绘制前多少个样本，避免图太拥挤
    """

    y_true = y_true[:num_points]
    y_pred = y_pred[:num_points]

    plt.figure(figsize=(10, 5))

    plt.plot(
        y_true,
        label="True"
    )

    plt.plot(
        y_pred,
        label="Predicted"
    )

    plt.xlabel("Sample")
    plt.ylabel("Occupancy")

    plt.title("Parking Occupancy Prediction")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.show()

    plt.close()