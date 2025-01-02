import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from Model import *
from train_model import *
from mamba_model import *
# Define parameters for the new differential equations
# c = 1.0  # Default value
b = 0.5  # Parameter b
a = 0.2  # Parameter a
np.random.seed(42)  # 固定随机种子

"""
    seq2seq, mamba, transformer, autofc
    python FHN_main.py 
"""
model_name = "mamba"

def derivative(X, t, c, b, a):
    x, y = X
    dxdt = c * (x + y - x**3 / 3)
    dydt = -1 / c * (x + b * y - a)
    return [dxdt, dydt]

def generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length):
    c = np.random.uniform(1.5, 5)  # 随机生成 c 值
    t = np.linspace(0, tmax, Nt + 1)
    X0 = [x0, y0]
    res = odeint(derivative, X0, t, args=(c, b, a))
    x, y = res.T

    data = np.stack([x, y], axis=1)
    
    # Normalize data
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = (data - data_min) / (data_max - data_min)

    x_data, y_data = [], []

    for i in range(len(data) - seq_length - pred_length):
        x_data.append(data[i : i + seq_length])
        y_data.append(data[i + seq_length : i + seq_length + pred_length])

    return np.array(x_data), np.array(y_data), t, data_min, data_max

if model_name == "seq2seq":
    # Main code to generate data, train Seq2Seq, and visualize results
    tmax = 30  # 缩短最大时间段为30
    Nt = 300  # 时间步数调整为300以增加分辨率
    seq_length = 150
    pred_length = 150

    # Generate data using the new system
    x0, y0 = 1.0, 0.0
    x_data, y_data, t, data_min, data_max = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize Seq2Seq model
    input_dim = 2
    hidden_dim = 64
    output_dim = 2
    model = LV_Seq2Seq(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    Seq2seq_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    x_test, y_test, t_test, _, _ = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)
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
    true_future_denorm = y_test_denorm.reshape(-1, 2)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 2)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_x = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_y = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])

    print(rmse)
    print(f"RMSE (x): {rmse_x:.4f}")
    print(f"RMSE (y): {rmse_y:.4f}")

elif model_name == "mamba":

    tmax = 30  # 缩短最大时间段为30
    Nt = 300  # 时间步数调整为300以增加分辨率
    seq_length = 150
    pred_length = 150

    # Generate data using the new system
    x0, y0 = 1.0, 0.0
    x_data, y_data, t, data_min, data_max = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # Define Mamba model arguments
    model_args = ModelArgs(
        vocab_size=2,  # 输入特征维度
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
    model = Mamba(model_args, hidden_dim)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    def train_model_mamba(model, criterion, optimizer, x_train, y_train, epochs):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            inputs = x_train
            targets = y_train

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    epochs = 200
    train_model_mamba(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    x_test, y_test, t_test, _, _ = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    pred = pred.numpy()
    y_test = y_test.numpy()
    x_test = x_test.numpy()


    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    # Compute RMSE
    # true_future_denorm = y_test_denorm.reshape(-1, 3)  # Flatten to 2D
    # pred_future_denorm = pred_denorm.reshape(-1, 3)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 2)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 2)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_x = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_y = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])

    print(rmse)
    print(f"RMSE (x): {rmse_x:.4f}")
    print(f"RMSE (y): {rmse_y:.4f}")


elif model_name == "autofc":
    # Main code to generate data, train AutoFC, and visualize results
    tmax = 30  # Maximum time
    Nt = 300  # Time steps
    seq_length = 150
    pred_length = 150

    # Generate data using the new system
    x0, y0 = 1.0, 0.0
    x_data, y_data, t, data_min, data_max = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize AutoFC model
    input_dim = 2
    hidden_dim = 64
    model = LV_AutoFC(input_dim, hidden_dim, seq_length, pred_length)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    def train_model(model, criterion, optimizer, x_train, y_train, epochs):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    epochs = 200
    train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    x_test, y_test, t_test, _, _ = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    pred = pred.numpy()
    y_test = y_test.numpy()

    # Compute RMSE
    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 2)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 2)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_x = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_y = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])

    print(rmse)
    print(f"RMSE (x): {rmse_x:.4f}")
    print(f"RMSE (y): {rmse_y:.4f}")


elif model_name == "transformer":
    # Transformer Model
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, seq_length, pred_length):
            super(TransformerModel, self).__init__()
            self.input_fc = nn.Linear(input_dim, d_model)
            self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length + pred_length, d_model))
            self.transformer = nn.Transformer(
                d_model=d_model, 
                nhead=nhead, 
                num_encoder_layers=num_encoder_layers, 
                num_decoder_layers=num_decoder_layers,
                batch_first=True
            )
            self.output_fc = nn.Linear(d_model, input_dim)

        def forward(self, x):
            batch_size, seq_length, input_dim = x.size()
            x_encoded = self.input_fc(x) + self.positional_encoding[:, :seq_length, :]

            # Create a mask for the decoder to prevent attending to future positions
            tgt_mask = self.transformer.generate_square_subsequent_mask(seq_length).to(x.device)
            
            tgt_input = torch.zeros(batch_size, pred_length, x_encoded.size(-1)).to(x.device)
            output = self.transformer(x_encoded, tgt_input, tgt_mask=tgt_mask)
            return self.output_fc(output)

    # Main code to generate data, train Transformer, and visualize results
    tmax = 30  # Maximum time
    Nt = 300  # Time steps
    seq_length = 150
    pred_length = 150

    # Generate data using the new system
    x0, y0 = 1.0, 0.0
    x_data, y_data, t, data_min, data_max = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)

    # Initialize Transformer model
    input_dim = 2
    d_model = 64
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, seq_length, pred_length)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    def train_model(model, criterion, optimizer, x_train, y_train, epochs):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    epochs = 200
    train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Generate test data
    x_test, y_test, t_test, _, _ = generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Predict and visualize results
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    # Denormalize predictions and true values
    def denormalize(data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    pred = pred.numpy()
    y_test = y_test.numpy()

    # Compute RMSE
    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    pred_denorm = denormalize(pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)
    x_test = denormalize(x_test, data_min, data_max)
    y_test = denormalize(y_test, data_min, data_max)

    # Denormalized RMSE
    true_future_denorm = y_test_denorm.reshape(-1, 2)  # Flatten to 2D
    pred_future_denorm = pred_denorm.reshape(-1, 2)

    rmse = calculate_rmse(true_future_denorm, pred_future_denorm)
    rmse_x = calculate_rmse(true_future_denorm[:, 0], pred_future_denorm[:, 0])
    rmse_y = calculate_rmse(true_future_denorm[:, 1], pred_future_denorm[:, 1])

    print(rmse)
    print(f"RMSE (x): {rmse_x:.4f}")
    print(f"RMSE (y): {rmse_y:.4f}")

# Plot predictions vs true values
# def plot_predictions(t, x_test, y_test, pred, seq_length, title="Seq2Seq Prediction"):
#     plt.figure(figsize=(10, 6))
#     observed = x_test[0].numpy()
#     true_future = y_test[0]
#     predicted_future = pred[0]

#     time_observed = t[:seq_length]
#     time_future = t[seq_length:seq_length + len(true_future)]

#     plt.plot(time_observed, observed[:, 0], label="x (Observed)", color="blue")
#     plt.plot(time_observed, observed[:, 1], label="y (Observed)", color="red")
#     plt.plot(time_future, true_future[:, 0], linestyle="--", label="x (True Future)", color="blue")
#     plt.plot(time_future, true_future[:, 1], linestyle="--", label="y (True Future)", color="red")
#     plt.plot(time_future, predicted_future[:, 0], linestyle="dotted", label="x (Predicted Future)", color="blue")
#     plt.plot(time_future, predicted_future[:, 1], linestyle="dotted", label="y (Predicted Future)", color="red")

#     plt.xlabel("Time")
#     plt.ylabel("Values")
#     plt.title(title)
#     plt.legend()
#     plt.show()

# plot_predictions(t_test, x_test, y_test_denorm, pred_denorm, seq_length)