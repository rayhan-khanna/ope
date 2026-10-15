import torch.nn as nn


class NaiveCF(nn.Module):
    """Bilinear collaborative-filtering scorer: a context projection times an item table.

    Input
    ------
    dim_context: int
        The size of the user context x.

    n_items: int
        The number of items, N.

    emb_dim: int
        The embedding size, D.

    """

    def __init__(self, dim_context, n_items, emb_dim):
        super().__init__()
        self.project_context = nn.Linear(dim_context, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

    def forward(self, context, item_ids):
        """Predicted reward of each (context, item) pair.

        Input
        ------
        context: torch.Tensor, shape (M, dim_context)
            The user context of each of the M pairs.

        item_ids: torch.Tensor, shape (M, )
            The item of each pair.

        Output
        ------
        score: torch.Tensor, shape (M, )
            <W x + b, e_a> for each pair.

        """
        return (self.project_context(context) * self.item_embedding(item_ids)).sum(dim=1)
