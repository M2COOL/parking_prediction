import torch.nn as nn

from .base_model import BaseModel


class LSTMModel(BaseModel):

    def __init__(
            self,
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
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

        # x:
        # [batch_size, seq_len, input_size]

        output, _ = self.lstm(x)

        # 最后一个时间步
        last_output = output[:, -1, :]

        y = self.fc(last_output)

        return y