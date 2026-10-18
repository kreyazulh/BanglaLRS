import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update. Drop-in for an existing nn.Linear."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)  # starts as identity (no change at init)

    def forward(self, x):
        return self.base(x) + self.scaling * self.b(self.a(self.drop(x)))


def inject_lora(module, target_names=("q_proj", "k_proj", "v_proj", "out_proj"),
                rank=8, alpha=16, dropout=0.0):
    """Recursively replace named nn.Linear attention projections with LoRALinear.
    fairseq MultiheadAttention exposes q_proj/k_proj/v_proj/out_proj as nn.Linear,
    so name matching works without PEFT. Returns count replaced."""
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            replaced += 1
        else:
            replaced += inject_lora(child, target_names, rank, alpha, dropout)
    return replaced


def trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]
