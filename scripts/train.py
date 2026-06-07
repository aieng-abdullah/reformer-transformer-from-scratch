import argparse
import json
import math
import os
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reformer import (
    CopyDataset,
    ReformerModel,
    evaluate_reformer,
    generate_text,
    save_checkpoint,
    train_reformer,
)


def load_config(path):
    with open(path) as f:
        return json.load(f)


def train_single(config, model_cfg, dataset_cfg, device):
    model = ReformerModel(
        vocab_size=model_cfg["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=model_cfg["n_heads"],
        bucket_size=config["bucket_size"],
        ffn_chunks=model_cfg["ffn_chunks"],
        max_seq_len=model_cfg["max_seq_len"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_dataset = CopyDataset(
        model_cfg["max_seq_len"],
        model_cfg["vocab_size"],
        size=dataset_cfg["dataset_size"],
    )
    val_dataset = CopyDataset(
        model_cfg["max_seq_len"],
        model_cfg["vocab_size"],
        size=max(1, dataset_cfg["dataset_size"] // 5),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=dataset_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=dataset_cfg["batch_size"])

    use_amp = dataset_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    loss_history = []
    for epoch in range(dataset_cfg["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_history.append(loss.item())

    val_loss, val_ppl = evaluate_reformer(model, val_loader, criterion, device)
    return model, val_loss, val_ppl, loss_history


def run_sweep(cfg):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    sweep_cfg = cfg["sweep"]

    sweep_keys = list(sweep_cfg.keys())
    sweep_values = list(sweep_cfg.values())

    results = []
    best_model = None
    best_loss = float("inf")
    best_config = None
    best_history = None

    for values in product(*sweep_values):
        config = dict(zip(sweep_keys, values))
        print(f"\nTraining config: {config}")

        model, val_loss, val_ppl, history = train_single(
            config, model_cfg, training_cfg, device
        )
        results.append((config, val_loss, val_ppl))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_config = config
            best_history = history

    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    for config, loss, ppl in results:
        print(f"  Config: {config}")
        print(f"  Val Loss: {loss:.4f}, Perplexity: {ppl:.4f}")
        print()

    print(f"Best config: {best_config}")
    print(f"Best val loss: {best_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(best_model, None, 0, "checkpoints/best_model.pt")

    return best_model, best_config, best_history


def run_single(cfg):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    config = {
        "dim": model_cfg["dim"],
        "n_layers": model_cfg["n_layers"],
        "bucket_size": model_cfg["bucket_size"],
        "learning_rate": training_cfg["learning_rate"],
    }

    print(f"Training config: {config}")
    model, val_loss, val_ppl, history = train_single(
        config, model_cfg, training_cfg, device
    )
    print(f"\nVal Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(model, None, 0, "checkpoints/model.pt")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Reformer model")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["sweep", "single"],
        default="single",
        help="Run hyperparameter sweep or single training run",
    )
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device

    if args.mode == "sweep":
        run_sweep(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
