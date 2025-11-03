# test_trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from training.trainer import Trainer

# ---- tiny dummy pieces ----
class DummyCaptionModel(nn.Module):
    def __init__(self, vocab_size=32, hidden=8):
        super().__init__()
        self.proj = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size
        self.hidden = hidden

    def forward(self, **batch):
        labels = batch["labels"]              # [B, T]
        B, T = labels.shape
        x = torch.randn(B, T, self.hidden, device=labels.device)
        logits = self.proj(x)                 # [B, T, V]
        return {"logits": logits}

    def generate(self, visual_embeddings, max_length=16, num_beams=1):
        # return one caption per sample
        B = visual_embeddings.shape[0]
        return ["a dummy caption"] * B

    def decode(self, ids):
        return "decoded"

class ToyTrainDS(Dataset):
    def __init__(self, n=16, seq_len=6, vocab=32):
        self.n, self.seq_len, self.vocab = n, seq_len, vocab
    def __len__(self): return self.n
    def __getitem__(self, idx):
        labels = torch.randint(0, self.vocab, (self.seq_len,))
        return {"labels": labels}

class ToyValDS(Dataset):
    def __init__(self, n=8, emb_dim=16):
        self.n, self.emb_dim = n, emb_dim
    def __len__(self): return self.n
    def __getitem__(self, idx):
        vemb = torch.randn(self.emb_dim)          # [D]
        refs = ["reference caption"]
        return {"visual_embeddings": vemb, "references": refs}

class DummyEvaluator:
    def evaluate(self, predictions, references):
        return {"bleu": 0.12, "cider": 0.34}

# ---- minimal config from the YAML shape ----
CFG = {
    "training": {
        "learning_rate": 1e-4,
        "num_epochs": 2,
        "warmup_steps": 1,
        "max_length": 16,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "weight_decay": 0.0,
        "grad_accum_steps": 2,
        "max_norm": 1.0,
        "num_beams": 2,
        "save_top_k": 1,
        "pad_token_id": -100,
        "amp": False,
        "log_every": 10,
    },
    "evaluation": {
        "metrics": ["bleu", "cider"],
        "main_metric": "cider",
        "eval_frequency": 1,
    },
}

# ---- run a tiny smoke train ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyCaptionModel().to(device)

    train_loader = DataLoader(ToyTrainDS(), batch_size=4, shuffle=False)
    val_loader   = DataLoader(ToyValDS(),   batch_size=4, shuffle=False)

    trainer = Trainer(model, train_loader, val_loader, CFG, device)
    evaluator = DummyEvaluator()

    os.makedirs("ckpts_simple", exist_ok=True)
    trainer.train(evaluator, save_dir="ckpts_simple")

    print("Done. Checkpoints:", [p for p in os.listdir("ckpts_simple") if p.endswith(".pt")])