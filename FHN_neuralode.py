import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
import torchdiffeq

# Define parameters for the new differential equations
b = 0.5  # Parameter b
a = 0.2  # Parameter a
np.random.seed(42)  # Fixed random seed

def derivative(X, t, c, b, a):
    x, y = X
    dxdt = c * (x + y - x**3 / 3)
    dydt = -1 / c * (x + b * y - a)
    return [dxdt, dydt]

def generate_new_data(x0, y0, tmax, Nt, seq_length, pred_length):
    c = np.random.uniform(1.5, 5)  # Random c value
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

# Neural ODE Model
class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, ode_func, seq_length, pred_length):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.seq_length = seq_length
        self.pred_length = pred_length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        t = torch.linspace(0, self.pred_length, self.pred_length).to(x.device)
        y0 = x[:, -1, :]
        pred = torchdiffeq.odeint(self.ode_func, y0, t, method='rk4')
        return pred.permute(1, 0, 2)  # Rearrange dimensions to match output shape

# Main code to generate data, train Neural ODE, and visualize results
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

# Initialize Neural ODE model
input_dim = 2
hidden_dim = 64
ode_func = ODEFunc(input_dim, hidden_dim)
model = NeuralODE(ode_func, seq_length, pred_length)
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
# def plot_predictions(t, x_test, y_test, pred, seq_length, title="Neural ODE Prediction"):
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