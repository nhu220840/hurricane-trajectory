# src/models.py

import math
import torch
import torch.nn as nn

# ------------------- PyTorch LSTM Forecaster -------------------
# (This class is kept unchanged from the original models.py file)
class LSTMForecaster(nn.Module):
    """
    Standard PyTorch LSTM, predicts (Δlat, Δlon) => output_size=2
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


# ------------------- Scratch LSTM (FROM NOTEBOOK REbuildLSTM.ipynb) -------------------
# (The two classes below are copied from cells 8 and 10 of the notebook)

class _ManualLSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        # U*: input->hidden (with bias)
        self.Uf = nn.Linear(in_dim, hidden, bias=True)
        self.Ui = nn.Linear(in_dim, hidden, bias=True)
        self.Uo = nn.Linear(in_dim, hidden, bias=True)
        self.Ug = nn.Linear(in_dim, hidden, bias=True)
        # W*: hidden->hidden (no bias)
        self.Wf = nn.Linear(hidden, hidden, bias=False)
        self.Wi = nn.Linear(hidden, hidden, bias=False)
        self.Wo = nn.Linear(hidden, hidden, bias=False)
        self.Wg = nn.Linear(hidden, hidden, bias=False)

        # Initialization
        for lin in [self.Uf, self.Ui, self.Uo, self.Ug]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in [self.Wf, self.Wi, self.Wo, self.Wg]:
            nn.init.orthogonal_(lin.weight)
        # Positive forget bias to encourage "remembering" at the start
        with torch.no_grad():
            self.Uf.bias.fill_(1.0)

    def forward(self, x_t, h_prev, c_prev):
        f_t = torch.sigmoid(self.Uf(x_t) + self.Wf(h_prev))
        i_t = torch.sigmoid(self.Ui(x_t) + self.Wi(h_prev))
        o_t = torch.sigmoid(self.Uo(x_t) + self.Wo(h_prev))
        g_t = torch.tanh(   self.Ug(x_t) + self.Wg(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMFromScratchForecaster(nn.Module):
    """
    Stacked LSTM from _ManualLSTMCell. Get h at the last timestep -> Linear head to out_dim.
    (Logic from cell 10 - REbuildLSTM.ipynb)
    """
    def __init__(self, in_dim, hidden=20, num_layers=2, out_dim=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        # Rename 'input_size' to 'in_dim' to match notebook
        self.dropout_p = dropout if num_layers > 1 else 0.0

        self.cells = nn.ModuleList([
            _ManualLSTMCell(in_dim if l == 0 else hidden, hidden)
            for l in range(num_layers)
        ])
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.head = nn.Linear(hidden, out_dim)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, T, F]
        B, T, _ = x.size()
        device = x.device
        dtype = x.dtype

        hs = [torch.zeros(B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros(B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h_l, c_l = cell(inp, hs[l], cs[l])
                hs[l], cs[l] = h_l, c_l
                # dropout between layers (not applied to the last layer)
                inp = self.dropout(h_l) if (l < self.num_layers - 1) else h_l

        last_h = hs[-1]          # [B, H] at the last timestep
        return self.head(last_h) # [B, out_dim]