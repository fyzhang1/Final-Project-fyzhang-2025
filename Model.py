import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# Seq2Seq Model
class LV_Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LV_Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, future_steps):
        batch_size, seq_len, _ = x.size()
        _, (hidden, cell) = self.encoder(x)

        decoder_input = x[:, -1:, :]  # Last time step as input
        outputs = []
        for _ in range(future_steps):
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(output)
            outputs.append(output)
            decoder_input = output

        return torch.cat(outputs, dim=1)


# 2. Define sin Seq2Seq Model
class sin_Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(sin_Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, future_steps):
        batch_size, seq_len, _ = x.size()
        _, (hidden, cell) = self.encoder(x)

        decoder_input = x[:, -1:, :]  # Use the last input as the first decoder input
        outputs = []

        for _ in range(future_steps):
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(output)
            outputs.append(output)
            decoder_input = output

        return torch.cat(outputs, dim=1)


# 2. Define Auto-FC Model
class sin_AutoFCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, pred_length):
        super(sin_AutoFCModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim * seq_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * pred_length)
        )
        self.seq_length = seq_length
        self.pred_length = pred_length

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        output = self.fc_layers(x)
        output = output.view(batch_size, self.pred_length, -1)
        return output


# 2. Define Transformer Model
class sin_TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, seq_length, pred_length):
        super(sin_TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.seq_length = seq_length
        self.pred_length = pred_length

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.fc_in(x).permute(1, 0, 2)  # Convert to (seq_len, batch_size, hidden_dim)
        transformer_out = self.transformer(x)  # Pass through Transformer
        transformer_out = transformer_out.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, hidden_dim)
        output = self.fc_out(transformer_out[:, -self.pred_length:, :])  # Predict the future sequence
        return output


# 2. Define Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, x):
        return self.net(x)

class sin_NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, pred_length):
        super(sin_NeuralODE, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.seq_length = seq_length
        self.pred_length = pred_length

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_layer(x.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)
        t = torch.linspace(0, self.pred_length - 1, self.pred_length).to(x.device)
        ode_out = odeint(self.ode_func, x[:, -1, :], t).permute(1, 0, 2)
        return self.output_layer(ode_out)


class LV_NeuralODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(LV_NeuralODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入维度为 2
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # 输出维度为 2
        )

    def forward(self, t, x):
        return self.net(x)

class LV_NeuralODEModel(nn.Module):
    def __init__(self, func, seq_length, pred_length):
        super(LV_NeuralODEModel, self).__init__()
        self.func = func
        self.seq_length = seq_length
        self.pred_length = pred_length

    def forward(self, x):
        batch_size = x.size(0)
        t = torch.linspace(0, 1, self.seq_length + self.pred_length).to(x.device)
        outputs = []
        for i in range(batch_size):
            x0 = x[i, 0]  # Initial condition
            pred = odeint(self.func, x0, t)  # Solve ODE
            outputs.append(pred[-self.pred_length:])  # Only take the prediction part
        return torch.stack(outputs)


class LV_AutoFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, pred_length):
        super(LV_AutoFC, self).__init__()
        self.fc_in = nn.Linear(input_dim * seq_length, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim * pred_length)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        x = self.relu(self.fc_in(x))
        x = self.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        x = x.view(batch_size, self.pred_length, -1)  # Reshape to (batch_size, pred_length, input_dim)
        return x