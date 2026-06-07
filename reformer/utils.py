import torch


def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {filepath}, resuming from epoch {epoch}")
    return epoch
