import torch.nn as nn


class NaiveCF(nn.Module):
    """Bilinear collaborative-filtering scorer: a context projection times an item table."""

    def __init__(self, dim_context, n_items, emb_dim):
        super().__init__()
        self.project_context = nn.Linear(dim_context, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

    def forward(self, context, item_ids):
        return (self.project_context(context) * self.item_embedding(item_ids)).sum(dim=1)
