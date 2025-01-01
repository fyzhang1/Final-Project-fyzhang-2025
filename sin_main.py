from mamba_model import *
from gen_data import *
from train_model import *
from Model import *
from tool import *
# sin generate data

model_name = "transformer"

if __name__ == "__main__":

    if model_name == "mamba":
        # Generate data
        seq_length = 50
        pred_length = 50
        num_samples = 1000
        x_data, y_data = generate_data(seq_length, pred_length, num_samples)

        # Split data into train and test
        train_size = int(0.8 * num_samples)
        x_train = torch.tensor(x_data[:train_size], dtype=torch.float32).unsqueeze(-1).to(device)
        y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).unsqueeze(-1).to(device)
        x_test = torch.tensor(x_data[train_size:], dtype=torch.float32).unsqueeze(-1).to(device)
        y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).unsqueeze(-1).to(device)

        # Define model arguments
        model_args = ModelArgs(
        vocab_size=1,      # 输入嵌入的维度
        d_model=64,        # 模型隐藏层维度
        d_inner=128,       # 内部隐藏维度
        n_layer=4,         # 堆叠的残差块层数
        seq_in=50,         # 输入序列长度
        bias=True,         # 是否在线性层中使用偏置
        conv_bias=True,    # 是否在卷积层中使用偏置
        d_conv=3,          # 卷积核大小
        dt_rank=8,         # 动态时间维度的秩
        d_state=16         # 状态空间的维度
    )

        hidden_dim = 128        # 中间隐藏维度
        input_dim = 64          # 图卷积的输入维度
        output_dim = 64         # 图卷积的输出维度
        embed_dim = 16          # 嵌入维度
        cheb_k = 3              # 图卷积的切比雪夫阶数

        args = ModelArgs()
        hid = 128

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Initialize model, criterion, and optimizer
        model = Mamba(args, hid).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 200
        mamba_train_model(model, criterion, optimizer, x_train, y_train, epochs)

        # Predict and visualize
        predictions = predict_and_visualize(model, x_test, y_test, seq_length, pred_length)


        rmse = calculate_rmse(y_test, predictions)

        print(f"RMSE for Interpolation: {rmse:.4f}")
    
    elif model_name == "seq2seq":
        # 4. Main code
        seq_length = 30
        pred_length = 30
        num_samples = 1000

        # Generate data
        x_train_interp, y_train_interp = generate_data(seq_length, pred_length, num_samples, mode="interpolation")
        x_train_extrap, y_train_extrap = generate_data(seq_length, pred_length, num_samples, mode="extrapolation")

        # Convert data to PyTorch tensors
        x_train_interp = torch.tensor(x_train_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_interp = torch.tensor(y_train_interp, dtype=torch.float32).unsqueeze(-1)

        x_train_extrap = torch.tensor(x_train_extrap, dtype=torch.float32).unsqueeze(-1)
        y_train_extrap = torch.tensor(y_train_extrap, dtype=torch.float32).unsqueeze(-1)

        # Initialize model, loss, and optimizer
        input_dim = 1
        hidden_dim = 64
        output_dim = 1
        num_layers = 2
        epochs = 200

        model_interp = sin_Seq2SeqModel(input_dim, hidden_dim, output_dim, num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_interp.parameters(), lr=0.001)

        # Train interpolation model
        Seq2seq_train_model(model_interp, criterion, optimizer, x_train_interp, y_train_interp, epochs)

        # model_extrap = Seq2SeqModel(input_dim, hidden_dim, output_dim, num_layers)
        # optimizer = optim.Adam(model_extrap.parameters(), lr=0.001)

        # Train extrapolation model
        # train_model(model_extrap, criterion, optimizer, x_train_extrap, y_train_extrap, epochs)

        # Generate test data
        x_test_interp, y_test_interp = generate_data(seq_length, pred_length, 1, mode="interpolation")
        # x_test_extrap, y_test_extrap = generate_data(seq_length, pred_length, 1, mode="extrapolation")

        x_test_interp = torch.tensor(x_test_interp, dtype=torch.float32).unsqueeze(-1)
        y_test_interp = torch.tensor(y_test_interp, dtype=torch.float32).unsqueeze(-1)

        # x_test_extrap = torch.tensor(x_test_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_test_extrap = torch.tensor(y_test_extrap, dtype=torch.float32).unsqueeze(-1)

        # Predict and visualize results
        pred_interp = evaluate_model(model_interp, x_test_interp)

    elif model_name == "transformer":
        seq_length = 30
        pred_length = 30
        num_samples = 1000

        # Generate data for different modes
        x_train_in_interp, y_train_in_interp = generate_data(seq_length, pred_length, num_samples, mode="in_domain_interp")
        # x_train_in_extrap, y_train_in_extrap = generate_data(seq_length, pred_length, num_samples, mode="in_domain_extrap")
        x_train_out_interp, y_train_out_interp = generate_data(seq_length, pred_length, num_samples, mode="out_domain_interp")
        # x_train_out_extrap, y_train_out_extrap = generate_data(seq_length, pred_length, num_samples, mode="out_domain_extrap")

        # Convert data to PyTorch tensors
        x_train_in_interp = torch.tensor(x_train_in_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_in_interp = torch.tensor(y_train_in_interp, dtype=torch.float32).unsqueeze(-1)

        # x_train_in_extrap = torch.tensor(x_train_in_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_train_in_extrap = torch.tensor(y_train_in_extrap, dtype=torch.float32).unsqueeze(-1)

        x_train_out_interp = torch.tensor(x_train_out_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_out_interp = torch.tensor(y_train_out_interp, dtype=torch.float32).unsqueeze(-1)

        # x_train_out_extrap = torch.tensor(x_train_out_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_train_out_extrap = torch.tensor(y_train_out_extrap, dtype=torch.float32).unsqueeze(-1)

        # Initialize model, loss, and optimizer
        input_dim = 1
        hidden_dim = 64
        num_heads = 4
        num_layers = 2
        epochs = 200

        model = sin_TransformerModel(input_dim, hidden_dim, num_heads, num_layers, seq_length, pred_length)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train a model (e.g., for in-domain interpolation)
        transformer_train_model(model, criterion, optimizer, x_train_in_interp, y_train_in_interp, epochs)

        # Generate test data for in-domain interpolation
        x_test_in_interp, y_test_in_interp = generate_data(seq_length, pred_length, 1, mode="in_domain_interp")
        x_test_in_interp = torch.tensor(x_test_in_interp, dtype=torch.float32).unsqueeze(-1)
        y_test_in_interp = torch.tensor(y_test_in_interp, dtype=torch.float32).unsqueeze(-1)

        # Predict and visualize results
        pred_in_interp = evaluate_model(model, x_test_in_interp)

        # Calculate RMSE
        def calculate_rmse(y_true, y_pred):
            return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

        rmse_in_interp = calculate_rmse(y_test_in_interp, pred_in_interp)
        print(f"RMSE for In-Domain Interpolation: {rmse_in_interp:.4f}")


        plot_results(x_test_in_interp, y_test_in_interp, pred_in_interp, "transformer In-Domain Interpolation")

    elif model_name == "autofc":
        seq_length = 30
        pred_length = 30
        num_samples = 1000

        # Generate data for different modes
        x_train_in_interp, y_train_in_interp = generate_data(seq_length, pred_length, num_samples, mode="in_domain_interp")
        # x_train_in_extrap, y_train_in_extrap = generate_data(seq_length, pred_length, num_samples, mode="in_domain_extrap")
        x_train_out_interp, y_train_out_interp = generate_data(seq_length, pred_length, num_samples, mode="out_domain_interp")
        # x_train_out_extrap, y_train_out_extrap = generate_data(seq_length, pred_length, num_samples, mode="out_domain_extrap")

        # Convert data to PyTorch tensors
        x_train_in_interp = torch.tensor(x_train_in_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_in_interp = torch.tensor(y_train_in_interp, dtype=torch.float32).unsqueeze(-1)

        # x_train_in_extrap = torch.tensor(x_train_in_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_train_in_extrap = torch.tensor(y_train_in_extrap, dtype=torch.float32).unsqueeze(-1)

        x_train_out_interp = torch.tensor(x_train_out_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_out_interp = torch.tensor(y_train_out_interp, dtype=torch.float32).unsqueeze(-1)

        # x_train_out_extrap = torch.tensor(x_train_out_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_train_out_extrap = torch.tensor(y_train_out_extrap, dtype=torch.float32).unsqueeze(-1)

        # Initialize model, loss, and optimizer
        input_dim = 1
        hidden_dim = 64
        output_dim = 1
        epochs = 200

        model = sin_AutoFCModel(input_dim, hidden_dim, output_dim, seq_length, pred_length)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train a model (e.g., for in-domain interpolation)
        autofc_train_model(model, criterion, optimizer, x_train_in_interp, y_train_in_interp, epochs)

        # Generate test data for in-domain interpolation
        x_test_in_interp, y_test_in_interp = generate_data(seq_length, pred_length, 1, mode="in_domain_interp")
        x_test_in_interp = torch.tensor(x_test_in_interp, dtype=torch.float32).unsqueeze(-1)
        y_test_in_interp = torch.tensor(y_test_in_interp, dtype=torch.float32).unsqueeze(-1)

        # Predict and visualize results
        pred_in_interp = evaluate_model(model, x_test_in_interp)

        # Calculate RMSE
        def calculate_rmse(y_true, y_pred):
            return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

        rmse_in_interp = calculate_rmse(y_test_in_interp, pred_in_interp)
        print(f"RMSE for In-Domain Interpolation: {rmse_in_interp:.4f}")

        plot_results(x_test_in_interp, y_test_in_interp, pred_in_interp, "AutoFC In-Domain Interpolation")
    

    elif model_name == "neuralode":
        seq_length = 30
        pred_length = 30
        num_samples = 1000

        # Generate data for different modes
        x_train_in_interp, y_train_in_interp = generate_data(seq_length, pred_length, num_samples, mode="in_domain_interp")
        # x_train_in_extrap, y_train_in_extrap = generate_data(seq_length, pred_length, num_samples, mode="in_domain_extrap")
        x_train_out_interp, y_train_out_interp = generate_data(seq_length, pred_length, num_samples, mode="out_domain_interp")
        # x_train_out_extrap, y_train_out_extrap = generate_data(seq_length, pred_length, num_samples, mode="out_domain_extrap")

        # Convert data to PyTorch tensors
        x_train_in_interp = torch.tensor(x_train_in_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_in_interp = torch.tensor(y_train_in_interp, dtype=torch.float32).unsqueeze(-1)

        # x_train_in_extrap = torch.tensor(x_train_in_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_train_in_extrap = torch.tensor(y_train_in_extrap, dtype=torch.float32).unsqueeze(-1)

        x_train_out_interp = torch.tensor(x_train_out_interp, dtype=torch.float32).unsqueeze(-1)
        y_train_out_interp = torch.tensor(y_train_out_interp, dtype=torch.float32).unsqueeze(-1)

        # x_train_out_extrap = torch.tensor(x_train_out_extrap, dtype=torch.float32).unsqueeze(-1)
        # y_train_out_extrap = torch.tensor(y_train_out_extrap, dtype=torch.float32).unsqueeze(-1)

        # Initialize model, loss, and optimizer
        input_dim = 1
        hidden_dim = 64
        output_dim = 1
        epochs = 200

        model = sin_NeuralODE(input_dim, hidden_dim, output_dim, seq_length, pred_length)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train a model (e.g., for in-domain interpolation)
        neuralode_train_model(model, criterion, optimizer, x_train_in_interp, y_train_in_interp, epochs)

        # Generate test data for in-domain interpolation
        x_test_in_interp, y_test_in_interp = generate_data(seq_length, pred_length, 1, mode="in_domain_interp")
        x_test_in_interp = torch.tensor(x_test_in_interp, dtype=torch.float32).unsqueeze(-1)
        y_test_in_interp = torch.tensor(y_test_in_interp, dtype=torch.float32).unsqueeze(-1)

        # Predict and visualize results
        pred_in_interp = evaluate_model(model, x_test_in_interp)

        # Calculate RMSE
        def calculate_rmse(y_true, y_pred):
            return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

        rmse_in_interp = calculate_rmse(y_test_in_interp, pred_in_interp)
        print(f"RMSE for In-Domain Interpolation: {rmse_in_interp:.4f}")

        plot_results(x_test_in_interp, y_test_in_interp, pred_in_interp, "ODE In-Domain Interpolation")