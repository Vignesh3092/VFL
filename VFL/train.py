import torch
import torch.nn as nn
import torch.optim as optim
from defense import apply_kmeans_defense
from data_loader import load_mnist
from models import ClientModel, ServerModel

def train_and_evaluate(poisoned_clients=1, num_clients=5):
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 5
    hidden_size = 128
    output_size = 10

    train_dataset, test_dataset = load_mnist()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = [ClientModel().to(device) for _ in range(num_clients)]
    split_width = 28 // num_clients
    dummy_input = torch.randn(1, 1, 28, split_width).to(device)
    embedding_size = clients[0](dummy_input).shape[1]
    server = ServerModel(embedding_size * num_clients, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(client.parameters(), lr=learning_rate) for client in clients]
    optimizer_server = optim.Adam(server.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for client in clients:
            client.train()
        server.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            raw_embeddings = []

            for i, client in enumerate(clients):
                part = images[:, :, :, i * split_width:(i + 1) * split_width]
                emb = client(part)
                raw_embeddings.append(emb)

            selected_embeddings = apply_kmeans_defense(raw_embeddings)
            combined_embeddings = torch.cat(selected_embeddings, dim=1)

            outputs = server(combined_embeddings)
            loss = criterion(outputs, labels)

            for optimizer in optimizers:
                optimizer.zero_grad()
            optimizer_server.zero_grad()
            loss.backward()

            for i, client in enumerate(clients):
                if i < poisoned_clients:
                    with torch.no_grad():
                        for param in client.parameters():
                            param.grad += torch.randn_like(param.grad) * 0.5
                torch.nn.utils.clip_grad_norm_(client.parameters(), 1.0)

            torch.nn.utils.clip_grad_norm_(server.parameters(), 1.0)

            for optimizer in optimizers:
                optimizer.step()
            optimizer_server.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    return evaluate_model(clients, server, test_loader, split_width, device)

def evaluate_model(clients, server, test_loader, split_width, device):
    for client in clients:
        client.eval()
    server.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = []

            for i, client in enumerate(clients):
                part = images[:, :, :, i * split_width:(i + 1) * split_width]
                embeddings.append(client(part))

            combined_embeddings = torch.cat(embeddings, dim=1)
            outputs = server(combined_embeddings)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100
