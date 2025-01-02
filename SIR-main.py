import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from mamba_model import *
from train_model import *
from Model import *
from gen_data import *
from mamba_model import *


# 计算 RMSE
def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


"""
    seq2seq, mamba, transformer, autofc, neuralode
    python SIR-main.py 
"""

model_name = "transformer"

# # SIR model parameters
# N = 350  # Total number of individuals
# I0, R0 = 1., 0  # Initial number of infected and recovered individuals
# S0 = N - I0 - R0  # Initial number of susceptible individuals
# beta, gamma = 0.4, 0.1  # Contact rate and mean recovery rate

# # Define the SIR derivatives
# def sir_derivative(X, t):
#     S, I, R = X
#     dotS = -beta * S * I / N
#     dotI = beta * S * I / N - gamma * I
#     dotR = gamma * I
#     return np.array([dotS, dotI, dotR])

# # Generate SIR data
# def generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length):
#     t = np.linspace(0, tmax, Nt + 1)
#     X0 = [S0, I0, R0]
#     res = odeint(sir_derivative, X0, t)
#     S, I, R = res.T

#     data = np.stack([S, I, R], axis=1)
    
#     # Normalize data
#     data_min = data.min(axis=0)
#     data_max = data.max(axis=0)
#     data = (data - data_min) / (data_max - data_min)

#     x_data, y_data = [], []

#     for i in range(len(data) - seq_length - pred_length):
#         x_data.append(data[i : i + seq_length])
#         y_data.append(data[i + seq_length : i + seq_length + pred_length])

#     return np.array(x_data), np.array(y_data), t, data_min, data_max

# Seq2Seq Model
# class Seq2Seq(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(Seq2Seq, self).__init__()
#         self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, future_steps):
#         batch_size, seq_len, _ = x.size()
#         _, (hidden, cell) = self.encoder(x)

#         decoder_input = x[:, -1:, :]  # Last time step as input
#         outputs = []
#         for _ in range(future_steps):
#             output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
#             output = self.fc(output)
#             outputs.append(output)
#             decoder_input = output

#         return torch.cat(outputs, dim=1)


    # Plot predictions vs true values
    # plot_sir_predictions(t_test, x_test.cpu(), y_test_denorm, pred_denorm, seq_length)



def evaluate_model(model, x_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    return predictions

if model_name == "seq2seq":
    # Main code to generate data, train Seq2Seq, and visualize results
    tmax = 160
    Nt = 160
    seq_length = 50
    pred_length = 50

    # Generate SIR data
    x_data, y_data, t, data_min, data_max = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize Seq2Seq model
    input_dim = 3
    hidden_dim = 64
    output_dim = 3
    model = LV_Seq2Seq(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    Seq2seq_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    x_test, y_test, t_test, _, _ = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test, y_test.size(1))

    # Denormalize predictions and true values
    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    pred = pred.numpy()
    y_test = y_test.numpy()

    # 计算 RMSE
    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 3)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 3)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_susceptible = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_infected = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])
    rmse_recovered = calculate_rmse(true_future_denorm[:, 2], pred_future_denorm[:, 2])

    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE (Susceptible): {rmse_susceptible:.4f}")
    print(f"RMSE (Infected): {rmse_infected:.4f}")
    print(f"RMSE (Recovered): {rmse_recovered:.4f}")



elif model_name == "mamba":

    tmax = 160
    Nt = 160
    seq_length = 50
    pred_length = 50
    # Initialize Mamba model

        # Generate SIR data
    x_data, y_data, t, data_min, data_max = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define Mamba model arguments
    model_args = ModelArgs(
        vocab_size=3,  # 输入特征维度 S, I, R
        d_model=64,
        d_inner=128,
        n_layer=4,
        seq_in=seq_length,
        bias=True,
        conv_bias=True,
        d_conv=3,
        dt_rank=8,
        d_state=16,
    )

    hidden_dim = 128
    model = Mamba(model_args, hidden_dim).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    def train_model(model, criterion, optimizer, x_train, y_train, epochs):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            inputs = x_train.to(device)
            targets = y_train.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    epochs = 200
    train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    x_test, y_test, t_test, _, _ = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    pred = pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    x_test = x_test.cpu().numpy()


    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Compute RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 3)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 3)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_susceptible = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_infected = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])
    rmse_recovered = calculate_rmse(true_future_denorm[:, 2], pred_future_denorm[:, 2])

    print(rmse)
    print(f"RMSE (Susceptible): {rmse_susceptible:.4f}")
    print(f"RMSE (Infected): {rmse_infected:.4f}")
    print(f"RMSE (Recovered): {rmse_recovered:.4f}")



elif model_name == "autofc":
    # Main code to generate data, train Seq2Seq, and visualize results
    tmax = 160
    Nt = 160
    seq_length = 50
    pred_length = 50

    # Generate SIR data
    x_data, y_data, t, data_min, data_max = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)


    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)
    # Convert data to PyTorch tensors
    # x_train = torch.tensor(x_data, dtype=torch.float32)
    # y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize
    input_dim = 3
    hidden_dim = 128
    model = LV_AutoFC(input_dim, hidden_dim, seq_length, pred_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    autofc_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    # x_test, y_test, t_test, _, _ = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)
    # x_test = torch.tensor(x_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    # pred = pred.numpy()
    # y_test = y_test.numpy()

    pred = pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    x_test = x_test.cpu().numpy()

    # 计算 RMSE
    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 3)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 3)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_susceptible = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_infected = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])
    rmse_recovered = calculate_rmse(true_future_denorm[:, 2], pred_future_denorm[:, 2])

    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE (Susceptible): {rmse_susceptible:.4f}")
    print(f"RMSE (Infected): {rmse_infected:.4f}")
    print(f"RMSE (Recovered): {rmse_recovered:.4f}")


elif model_name == "transformer":
    # Main code to generate data, train Seq2Seq, and visualize results
    tmax = 160
    Nt = 160
    seq_length = 50
    pred_length = 50

    # Generate SIR data
    x_data, y_data, t, data_min, data_max = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)


    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)
    # Convert data to PyTorch tensors
    # x_train = torch.tensor(x_data, dtype=torch.float32)
    # y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize
    input_dim = 3
    hidden_dim = 128
    num_heads = 4
    num_layers = 3
    model = sin_TransformerModel(input_dim, hidden_dim, num_heads, num_layers, seq_length, pred_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    transformer_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    # x_test, y_test, t_test, _, _ = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)
    # x_test = torch.tensor(x_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    # pred = pred.numpy()
    # y_test = y_test.numpy()

    pred = pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    x_test = x_test.cpu().numpy()

    # 计算 RMSE
    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 3)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 3)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_susceptible = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_infected = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])
    rmse_recovered = calculate_rmse(true_future_denorm[:, 2], pred_future_denorm[:, 2])

    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE (Susceptible): {rmse_susceptible:.4f}")
    print(f"RMSE (Infected): {rmse_infected:.4f}")
    print(f"RMSE (Recovered): {rmse_recovered:.4f}")



elif model_name == "neuralode":
    # Main code to generate data, train Seq2Seq, and visualize results
    tmax = 160
    Nt = 160
    seq_length = 50
    pred_length = 50

    # Generate SIR data
    x_data, y_data, t, data_min, data_max = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)


    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)
    # Convert data to PyTorch tensors
    # x_train = torch.tensor(x_data, dtype=torch.float32)
    # y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize
    hidden_dim = 128
    func = LV_NeuralODEFunc(hidden_dim).to(device)
    model = LV_NeuralODEModel(func, seq_length, pred_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    neuralode_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    # x_test, y_test, t_test, _, _ = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)
    # x_test = torch.tensor(x_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    # pred = pred.numpy()
    # y_test = y_test.numpy()

    pred = pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    x_test = x_test.cpu().numpy()

    # 计算 RMSE
    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 3)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 3)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_susceptible = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_infected = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])
    rmse_recovered = calculate_rmse(true_future_denorm[:, 2], pred_future_denorm[:, 2])

    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE (Susceptible): {rmse_susceptible:.4f}")
    print(f"RMSE (Infected): {rmse_infected:.4f}")
    print(f"RMSE (Recovered): {rmse_recovered:.4f}")

# Plot predictions vs true values
# def plot_sir_predictions(t, x_test, y_test, pred, seq_length, title="SIR Seq2Seq Prediction"):
#     plt.figure(figsize=(10, 6))
#     observed = x_test[0].numpy()
#     true_future = y_test[0]
#     predicted_future = pred[0]

#     time_observed = t[:seq_length]
#     time_future = t[seq_length:seq_length + len(true_future)]

#     plt.plot(time_observed, observed[:, 0], label="Susceptible (Observed)", color="blue")
#     plt.plot(time_observed, observed[:, 1], label="Infected (Observed)", color="red")
#     plt.plot(time_observed, observed[:, 2], label="Recovered (Observed)", color="green")
#     plt.plot(time_future, true_future[:, 0], linestyle="--", label="Susceptible (True Future)", color="blue")
#     plt.plot(time_future, true_future[:, 1], linestyle="--", label="Infected (True Future)", color="red")
#     plt.plot(time_future, true_future[:, 2], linestyle="--", label="Recovered (True Future)", color="green")
#     plt.plot(time_future, predicted_future[:, 0], linestyle="dotted", label="Susceptible (Predicted Future)", color="blue")
#     plt.plot(time_future, predicted_future[:, 1], linestyle="dotted", label="Infected (Predicted Future)", color="red")
#     plt.plot(time_future, predicted_future[:, 2], linestyle="dotted", label="Recovered (Predicted Future)", color="green")

#     plt.xlabel("Time")
#     plt.ylabel("Population")
#     plt.title(title)
#     plt.legend()
#     plt.show()

# plot_sir_predictions(t_test, x_test, y_test_denorm, pred_denorm, seq_length)