import torch.nn as nn

class NaiveCF(nn.Module):
    def __init__(self, dim_context, n_items, emb_dim=32):
        super().__init__()
        self.project_context = nn.Linear(dim_context, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

    def forward(self, context, item_ids):
        context_vecs = self.project_context(context)  
        item_vecs = self.item_embedding(item_ids)
        return (context_vecs * item_vecs).sum(dim=1)