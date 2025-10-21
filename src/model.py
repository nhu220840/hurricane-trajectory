# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Lớp Attention."""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(hidden_states))
        # energy shape: (batch_size, seq_len, hidden_dim)
        attn_scores = torch.einsum("bsh,h->bs", energy, self.v)
        # attn_scores shape: (batch_size, seq_len)
        return F.softmax(attn_scores, dim=1)

class LSTMAttention(nn.Module):
    """Kiến trúc mạng LSTM với cơ chế Attention."""
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, dropout):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # Sử dụng LSTM 2 chiều để học ngữ cảnh tốt hơn
        )
        self.attention = Attention(hidden_dim * 2) # *2 vì là bidirectional
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)

        attn_weights = self.attention(lstm_out)
        # attn_weights shape: (batch, seq_len)

        context_vector = torch.einsum("bs,bsh->bh", attn_weights, lstm_out)
        # context_vector shape: (batch, hidden_dim * 2)

        return self.head(context_vector)