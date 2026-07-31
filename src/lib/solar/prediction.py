import torch
import torch.nn as nn
import math

class GHIPredictorLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=24):
        super(GHIPredictorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last timestep of the sequence
        out = out[:, -1, :]
        out = self.fc(out)
        out = torch.clamp(out, 0.0, 1.0)  # Bound output to [0, 1] range
        return out


class CausalConv1d(nn.Module):
    """
    A 1D causal convolution that pads only the left (past) so that 
    the output at timestep t depends only on inputs up to t.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        # Slicing off the right padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class GHIPredictorTCN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=3, output_size=24):
        super(GHIPredictorTCN, self).__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation_size = 2 ** i
            layers += [
                CausalConv1d(in_channels, hidden_size, kernel_size=3, dilation=dilation_size),
                nn.ReLU()
            ]
            in_channels = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2) # Conv1d expects (batch_size, channels, length)
        out = self.tcn(x)
        # Take the last element in the sequence
        out = out[:, :, -1]
        out = self.fc(out)
        out = torch.clamp(out, 0.0, 1.0)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class GHIPredictorTransformer(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=24):
        super(GHIPredictorTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=1000)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        out = self.transformer_encoder(x)
        
        # Take the output of the last timestep
        out = out[:, -1, :] 
        out = self.fc(out)
        out = torch.clamp(out, 0.0, 1.0)
        return out
