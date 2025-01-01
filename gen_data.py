import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Generate sine wave data based on the given modes
def generate_data(seq_length, pred_length, num_samples, mode="in_domain_interp"):
    x_data = []
    y_data = []

    for _ in range(num_samples):
        if "in_domain" in mode:
            omega = np.pi
            b = 0
        else:  # out-of-domain modes
            omega = np.random.uniform(0.5 * np.pi, 1.5 * np.pi)
            b = np.random.uniform(-np.pi, np.pi)

        if "interp" in mode:
            t_start = np.random.uniform(0, 10)
            t = np.linspace(t_start, t_start + (seq_length + pred_length) * 0.1, seq_length + pred_length)
        else:  # extrapolation
            t_start = np.random.uniform(20, 30)
            t = np.linspace(t_start, t_start + (seq_length + pred_length) * 0.1, seq_length + pred_length)

        x_seq = np.sin(omega * t[:seq_length] + b)
        y_seq = np.sin(omega * t[seq_length:] + b)

        x_data.append(x_seq)
        y_data.append(y_seq)

    return np.array(x_data), np.array(y_data)


# Define the Lotka-Volterra derivatives
def derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

# Generate Lotka-Volterra data
def generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length):
    t = np.linspace(0, tmax, Nt)
    X0 = [x0, y0]
    res = odeint(derivative, X0, t, args=(alpha, beta, delta, gamma))
    x, y = res.T

    data = np.stack([x, y], axis=1)
    x_data, y_data = [], []

    for i in range(len(data) - seq_length - pred_length):
        x_data.append(data[i : i + seq_length])
        y_data.append(data[i + seq_length : i + seq_length + pred_length])
    
    return np.array(x_data), np.array(y_data), t

