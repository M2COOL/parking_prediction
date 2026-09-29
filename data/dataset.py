import numpy as np
import torch
from torch.utils.data import Dataset


class ParkingDataset(Dataset):

    def __init__(self, num_samples=2000, seq_len=12):
        self.seq_len = seq_len

        t = np.arange(num_samples + seq_len)

        # 模拟停车占用率
        occupancy = (
                0.5
                + 0.25 * np.sin(2 * np.pi * t / 24)
                + 0.15 * np.sin(2 * np.pi * t / 12)
                + 0.05 * np.random.randn(len(t))
        )

        occupancy = np.clip(occupancy, 0, 1)

        X = []
        y = []

        for i in range(num_samples):
            X.append(
                occupancy[i:i + seq_len]
            )

            y.append(
                occupancy[i + seq_len]
            )

        self.X = torch.tensor(
            np.array(X),
            dtype=torch.float32
        ).unsqueeze(-1)

        self.y = torch.tensor(
            np.array(y),
            dtype=torch.float32
        ).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]