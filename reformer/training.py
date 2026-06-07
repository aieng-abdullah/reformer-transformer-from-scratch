import math

import torch


def train_reformer(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
    return loss_history


def evaluate_reformer(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_tokens += x.size(0) * x.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity
