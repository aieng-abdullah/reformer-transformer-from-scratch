import torch
from torch.utils.data import Dataset


class CopyDataset(Dataset):
    def __init__(self, seq_len, vocab_size, size=1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size, (self.seq_len,))
        y = x.clone()
        return x, y
