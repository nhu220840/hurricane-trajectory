# src/models.py
import torch
from torch import nn

# --- Model 1: Từ 'REbuildLSTM.ipynb' ---

class _ManualLSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        # ... (Sao chép y hệt class _ManualLSTMCell từ notebook) ...
        self.hidden = hidden
        self.Uf = nn.Linear(in_dim, hidden, bias=True)
        self.Ui = nn.Linear(in_dim, hidden, bias=True)
        self.Uo = nn.Linear(in_dim, hidden, bias=True)
        self.Ug = nn.Linear(in_dim, hidden, bias=True)
        self.Wf = nn.Linear(hidden, hidden, bias=False)
        self.Wi = nn.Linear(hidden, hidden, bias=False)
        self.Wo = nn.Linear(hidden, hidden, bias=False)
        self.Wg = nn.Linear(hidden, hidden, bias=False)
        for lin in [self.Uf, self.Ui, self.Uo, self.Ug]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in [self.Wf, self.Wi, self.Wo, self.Wg]:
            nn.init.orthogonal_(lin.weight)
        with torch.no_grad():
            self.Uf.bias.fill_(1.0)

    def forward(self, x_t, h_prev, c_prev):
        # ... (Sao chép y hệt hàm forward) ...
        f_t = torch.sigmoid(self.Uf(x_t) + self.Wf(h_prev))
        i_t = torch.sigmoid(self.Ui(x_t) + self.Wi(h_prev))
        o_t = torch.sigmoid(self.Uo(x_t) + self.Wo(h_prev))
        g_t = torch.tanh(self.Ug(x_t) + self.Wg(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMFromScratchForecaster(nn.Module):
    """ Model 1: LSTM tự xây dựng """

    def __init__(self, in_dim, hidden, num_layers, out_dim, dropout):
        super().__init__()
        # ... (Sao chép y hệt class LSTMFromScratchForecaster từ notebook) ...
        self.hidden = hidden
        self.num_layers = num_layers
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
        # ... (Sao chép y hệt hàm forward) ...
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
                inp = self.dropout(h_l) if (l < self.num_layers - 1) else h_l
        last_h = hs[-1]
        return self.head(last_h)


# --- Model 2: Từ 'scripts/train_model.py' ---

class LSTMForecaster(nn.Module):
    """ Model 2: LSTM có sẵn của PyTorch """

    def __init__(self, in_dim, hidden, num_layers, out_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]  # Lấy output của timestep cuối cùng
        return self.head(last_h)