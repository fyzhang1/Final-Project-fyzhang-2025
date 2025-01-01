import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from mamba_model import *
from gen_data import *
from train_model import *
from Model import *
from tool import *
#LV generate data

# Lotka-Volterra parameters
alpha = 1.0  # mortality rate due to predators
beta = 1.0
delta = 1.0
gamma = 1.0
x0 = 4.0
y0 = 2.0
tmax = 30.0
Nt = 1000

model_name = "Seq2Seq"

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


if model_name == "Seq2Seq":
    # Main code to generate data, train Seq2Seq, and visualize results
    seq_length = 50
    pred_length = 50

    # Generate Lotka-Volterra data
    x_data, y_data, t = generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length)

    # Convert data to PyTorch tensors
    # x_train = torch.tensor(x_data, dtype=torch.float32)
    # y_train = torch.tensor(y_data, dtype=torch.float32)

    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)

    # Initialize Seq2Seq model
    input_dim = 2
    hidden_dim = 64
    output_dim = 2
    model = LV_Seq2Seq(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # num_layers = 2
    epochs = 200
    # # Generate test data
    # x_test, y_test, t_test = generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length)
    # x_test = torch.tensor(x_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)
    Seq2seq_train_model(model, criterion, optimizer, x_train, y_train, epochs)
    # Predict and visualize
    # predictions = predict_and_visualize(model, x_test, y_test, seq_length, pred_length)

    with torch.no_grad():
        predictions = model(x_test, y_test.size(1))
    # Calculate RMSE
    rmse = calculate_rmse(y_test, predictions)
    print(f"RMSE for Lotka-Volterra Predictions: {rmse:.4f}")

    plot_lv_predictions(t, x_test.cpu(), y_test.cpu(), predictions.cpu(), seq_length, model_name)

    # Predict and visualize results


    # model.eval()
    # with torch.no_grad():
    #     pred = model(x_test, y_test.size(1))


    # rmse = calculate_rmse(y_test, predictions)
    # print(f"RMSE for Interpolation: {rmse:.4f}")

    # # plot_lv_predictions(t_test, x_test, y_test, pred, seq_length, model_name)
    # plot_lv_predictions(t, x_test.cpu(), y_test.cpu(), predictions.cpu(), seq_length, model_name)


elif model_name == "Mamba":
    # Generate Lotka-Volterra data
    seq_length = 50
    pred_length = 50
    x_data, y_data, t = generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length)

    # Split data into train and test
    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)

    # Define model arguments
    model_args = ModelArgs(
        vocab_size=2,  # Two input features: Prey and Predator
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

    # Initialize Mamba model
    model = Mamba(model_args, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 200
    mamba_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Predict and visualize
    predictions = predict_and_visualize(model, x_test, y_test, seq_length, pred_length)

    # Calculate RMSE
    rmse = calculate_rmse(y_test, predictions)
    print(f"RMSE for Lotka-Volterra Predictions: {rmse:.4f}")

    plot_lv_predictions(t, x_test.cpu(), y_test.cpu(), predictions.cpu(), seq_length, model_name)

elif model_name == "autofc":

    seq_length = 50
    pred_length = 50
    # Generate Lotka-Volterra data
    x_data, y_data, t = generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length)

    # Split data into train and test
    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)

    # Initialize Auto-FC model
    input_dim = 2  # Two features: Prey and Predator
    hidden_dim = 128
    model = LV_AutoFC(input_dim, hidden_dim, seq_length, pred_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 200
    autofc_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Predict and visualize
    predictions = predict_and_visualize(model, x_test, y_test, seq_length, pred_length)

    # Calculate RMSE
    rmse = calculate_rmse(y_test, predictions)
    print(f"RMSE for Lotka-Volterra Predictions: {rmse:.4f}")

    # x_test.cpu()
    # y_test.cpu()
    # predictions.cpu()

    # plot_lv_predictions(t, x_test, y_test, predictions, seq_length, model_name)
    plot_lv_predictions(t, x_test.cpu(), y_test.cpu(), predictions.cpu(), seq_length, model_name)

elif model_name == "transformer":
    seq_length = 50
    pred_length = 50
    # Generate Lotka-Volterra data
    x_data, y_data, t = generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length)

    # Split data into train and test
    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)

   
    input_dim = 2  # Two features: Prey and Predator
    hidden_dim = 128
    num_heads = 4
    num_layers = 3
    model = sin_TransformerModel(input_dim, hidden_dim, num_heads, num_layers, seq_length, pred_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 200
    transformer_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Predict and visualize
    predictions = predict_and_visualize(model, x_test, y_test, seq_length, pred_length)

    # Calculate RMSE
    rmse = calculate_rmse(y_test, predictions)
    print(f"RMSE for Lotka-Volterra Predictions: {rmse:.4f}")

    # x_test.cpu()
    # y_test.cpu()
    # predictions.cpu()

    # plot_lv_predictions(t, x_test, y_test, predictions, seq_length, model_name)
    plot_lv_predictions(t, x_test.cpu(), y_test.cpu(), predictions.cpu(), seq_length, model_name)

elif model_name == "neuralode":
    seq_length = 50
    pred_length = 50
    # Generate Lotka-Volterra data
    x_data, y_data, t = generate_lv_data(alpha, beta, delta, gamma, x0, y0, tmax, Nt, seq_length, pred_length)

    # Split data into train and test
    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).to(device)
    x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).to(device)

    hidden_dim = 128
    epochs = 200
    func = LV_NeuralODEFunc(hidden_dim).to(device)
    model = LV_NeuralODEModel(func, seq_length, pred_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    neuralode_train_model(model, criterion, optimizer, x_train, y_train, epochs)

    # Predict and visualize
    predictions = predict_and_visualize(model, x_test, y_test, seq_length, pred_length)

    # Calculate RMSE
    rmse = calculate_rmse(y_test, predictions)
    print(f"RMSE for Lotka-Volterra Predictions: {rmse:.4f}")

    plot_lv_predictions(t, x_test.cpu(), y_test.cpu(), predictions.cpu(), seq_length, model_name)