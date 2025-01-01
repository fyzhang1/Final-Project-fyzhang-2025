import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 4. Training and evaluation functions
def mamba_train_model(model, criterion, optimizer, x_train, y_train, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Move data to GPU
        inputs = x_train.to(device)
        targets = y_train.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# Train the model
def Seq2seq_train_model(model, criterion, optimizer, x_train, y_train, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x_train, y_train.size(1))
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def autofc_train_model(model, criterion, optimizer, x_train, y_train, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def transformer_train_model(model, criterion, optimizer, x_train, y_train, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def neuralode_train_model(model, criterion, optimizer, x_train, y_train, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")