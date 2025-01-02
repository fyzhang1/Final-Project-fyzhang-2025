import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# SIR model parameters
N = 350  # Total population
I0, R0 = 1., 0.  # Initial infected and recovered individuals
S0 = N - I0 - R0  # Initial susceptible individuals
beta, gamma = 0.4, 0.1  # Contact rate and recovery rate

# Define the SIR derivatives
from torchdiffeq import odeint as torch_odeint

def sir_derivative(t, X):
    S, I, R = X
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return torch.tensor([dS, dI, dR], dtype=torch.float32)

def generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length):
    t = torch.linspace(0, tmax, Nt + 1)  # 时间点
    X0 = torch.tensor([S0, I0, R0], dtype=torch.float32)  # 初始条件

    res = torch_odeint(sir_derivative, X0, t)  # 使用 torchdiffeq 解决 ODE
    S, I, R = res.T

    data = torch.stack([S, I, R], dim=1).cpu().numpy()  # 转换为 NumPy 数组
    
    # Normalize data
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = (data - data_min) / (data_max - data_min)

    x_data, y_data = [], []

    for i in range(len(data) - seq_length - pred_length):
        x_data.append(data[i : i + seq_length])
        y_data.append(data[i + seq_length : i + seq_length + pred_length])

    return np.array(x_data), np.array(y_data), t.cpu().numpy(), data_min, data_max


# Define Neural ODE Function
class SIR_NeuralODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(SIR_NeuralODEFunc, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Input features S, I, R
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output features S, I, R
        )

    def forward(self, t, x):
        return self.hidden_layer(x)

# Define Neural ODE Model
class SIR_NeuralODEModel(nn.Module):
    def __init__(self, func, seq_length, pred_length):
        super(SIR_NeuralODEModel, self).__init__()
        self.func = func
        self.seq_length = seq_length
        self.pred_length = pred_length

    def forward(self, x):
        batch_size = x.size(0)
        t = torch.linspace(0, 1, self.pred_length).to(x.device)  # Prediction time points
        preds = []
        for i in range(batch_size):
            x0 = x[i, -1]  # Last time step as initial condition
            pred = odeint(self.func, x0, t)  # Solve ODE
            preds.append(pred)
        preds = torch.stack(preds, dim=0)  # (batch_size, pred_length, 3)
        return preds

# Generate data
seq_length = 50
pred_length = 50
tmax = 160
Nt = 160

x_data, y_data, t, data_min, data_max = generate_sir_data(S0, I0, R0, tmax, Nt, seq_length, pred_length)

# Split data into training and testing sets
num_samples = len(x_data)
train_size = int(0.8 * num_samples)
x_train = torch.tensor(x_data[:train_size], dtype=torch.float32)
y_train = torch.tensor(y_data[:train_size], dtype=torch.float32)
x_test = torch.tensor(x_data[train_size:], dtype=torch.float32)
y_test = torch.tensor(y_data[train_size:], dtype=torch.float32)

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 128
func = SIR_NeuralODEFunc(hidden_dim).to(device)
model = SIR_NeuralODEModel(func, seq_length, pred_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_neural_ode(model, criterion, optimizer, x_train, y_train, epochs):
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
train_neural_ode(model, criterion, optimizer, x_train, y_train, epochs)

# Evaluate the model
model.eval()
with torch.no_grad():
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    pred = model(x_test)

# Denormalize predictions and true values
def denormalize(data, data_min, data_max):
    return data * (data_max - data_min) + data_min

pred = pred.cpu().numpy()
y_test = y_test.cpu().numpy()
pred_denorm = denormalize(pred, data_min, data_max)
y_test_denorm = denormalize(y_test, data_min, data_max)

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))

rmse = calculate_rmse(y_test_denorm, pred_denorm)
print(f"RMSE: {rmse:.4f}")

# Plot predictions vs true values
# def plot_sir_predictions(t, x_test, y_test, pred, seq_length, title="SIR Neural ODE Prediction"):
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

# plot_sir_predictions(t, x_test.cpu(), y_test_denorm, pred_denorm, seq_length)