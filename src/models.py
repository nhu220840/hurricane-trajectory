# src/models.py

import math
import torch
import torch.nn as nn

# ------------------- PyTorch LSTM Forecaster -------------------

class LSTMForecaster(nn.Module):
    """
    LSTM chuẩn PyTorch, dự đoán (Δlat, Δlon) => output_size=2
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)     # (B, N_IN, H)
        last = out[:, -1, :]      # (B, H)
        y = self.head(last)       # (B, 2)
        return y


# ------------------- Scratch LSTM Cell (min) -------------------

class LSTMCellScratch(nn.Module):
    """
    LSTMCell tự viết tối giản, dùng cho minh hoạ.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(4*hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4*hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4*hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x_t, hx):
        h, c = hx
        gates = (x_t @ self.weight_ih.T) + (h @ self.weight_hh.T) + self.bias
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LSTMFromScratchForecaster(nn.Module):
    """
    LSTM một lớp dùng LSTMCellScratch, head tuyến tính ra 2.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.0):
        super().__init__()
        assert num_layers == 1, "Bản scratch mẫu này hỗ trợ 1 layer cho gọn."
        self.cell = LSTMCellScratch(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        B, T, D = x.shape
        h = x.new_zeros(B, self.cell.hidden_size)
        c = x.new_zeros(B, self.cell.hidden_size)
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
        h = self.dropout(h)
        y = self.head(h)
        return y
