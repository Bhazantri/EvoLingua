
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributed import init_process_group
import os
from .model import EvoLingua
from .config import EvoLinguaConfig

class DummyDataset(Dataset):
    """Placeholder dataset."""
    def __init__(self, config: EvoLinguaConfig):
        self.config = config
    def __len__(self): return 1000
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,)),
            "attention_mask": torch.ones(self.config.max_seq_len),
            "labels": torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,))
        }

def train(model: EvoLingua, dataloader, optimizer, device, num_epochs: int = 1):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            main_output, mtp_outputs = model(input_ids, attention_mask)
            
            loss = F.cross_entropy(main_output.view(-1, model.config.vocab_size), labels.view(-1))
            for mtp_output in mtp_outputs:
                loss += F.cross_entropy(mtp_output.view(-1, model.config.vocab_size), labels.view(-1))
            loss /= (1 + model.config.mtp_depth)

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EvoLinguaConfig()
    model = EvoLingua(config).to(device)
    dataloader = DataLoader(DummyDataset(config), batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train(model, dataloader, optimizer, device)
