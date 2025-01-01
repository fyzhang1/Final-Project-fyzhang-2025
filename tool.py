import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Plot predictions vs true values
def plot_lv_predictions(t, x_test, y_test, pred, seq_length, model_name, title="Lotka-Volterra Prediction"):
    plt.figure(figsize=(10, 6))
    observed = x_test[0].numpy()
    true_future = y_test[0].numpy()
    predicted_future = pred[0].numpy()

    time_observed = t[:seq_length]
    time_future = t[seq_length:seq_length + len(true_future)]

    plt.plot(time_observed, observed[:, 0], label="Prey (Observed)", color="blue")
    plt.plot(time_observed, observed[:, 1], label="Predator (Observed)", color="red")
    plt.plot(time_future, true_future[:, 0], linestyle="--", label="Prey (True Future)", color="blue")
    plt.plot(time_future, true_future[:, 1], linestyle="--", label="Predator (True Future)", color="red")
    plt.plot(time_future, predicted_future[:, 0], linestyle="dotted", label="Prey (Predicted Future)", color="blue")
    plt.plot(time_future, predicted_future[:, 1], linestyle="dotted", label="Predator (Predicted Future)", color="red")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{model_name}_prediction_sample.png")
    plt.close()

def evaluate_model(model, x_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    return predictions

# 5. Prediction and Visualization
def predict_and_visualize(model, x_test, y_test, seq_length, pred_length):
    predictions = evaluate_model(model, x_test)

    # Visualize results
    # for i in range(3):  # Visualize first 3 samples
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(seq_length), x_test[i].cpu().numpy(), label="Input Sequence")
    #     plt.plot(range(seq_length, seq_length + pred_length), y_test[i].cpu().numpy(), label="True Output")
    #     plt.plot(range(seq_length, seq_length + pred_length), predictions[i].cpu().numpy(), label="Predicted Output", linestyle="--")
    #     plt.xlabel("Time Step")
    #     plt.ylabel("Value")
    #     plt.title(f"Sample {i + 1}: Prediction vs True Output")
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(f"prediction_sample_{i + 1}.png")
    #     plt.close()

    return predictions

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


# Visualization
def plot_results(x_test, y_test, pred, title):
    plt.figure()
    plt.plot(range(len(x_test[0])), x_test[0].squeeze().numpy(), label="Input")
    plt.plot(range(len(x_test[0]), len(x_test[0]) + len(y_test[0])), y_test[0].squeeze().numpy(), label="True")
    plt.plot(range(len(x_test[0]), len(x_test[0]) + len(pred[0])), pred[0].squeeze().numpy(), label="Predicted", linestyle="--")
    plt.title(title)
    plt.legend()
    plt.savefig(f"prediction_sample.png")
    plt.close()