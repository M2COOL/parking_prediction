import torch.nn as nn

from .base_model import BaseModel


class GRUModel(BaseModel):

    def __init__(
            self,
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        y = self.fc(last_output)

        return y