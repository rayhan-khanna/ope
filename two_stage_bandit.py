import torch
import torch.nn as nn

# candidate selection
class TwoTowerFirstStagePolicy(nn.Module):
    def __init__(self, dim_context, num_items, emb_dim, top_k):
        super().__init__()
        self.top_k = top_k  

        # context tower
        self.context_nn = nn.Sequential(
            nn.Linear(dim_context, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # item embeddings
        self.item_embeddings = nn.Embedding(num_items, emb_dim)

    def forward(self, x):
        context_repr = self.context_nn(x)  
        item_embs = self.item_embeddings.weight 

        # compute logits via inner product
        logits = torch.matmul(context_repr, item_embs.T) 
        probs = torch.softmax(logits, dim=1) 

        # select top-k 
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=1)

        return top_k_probs, top_k_indices

# action selection (softmax)
class SoftmaxSecondStagePolicy(nn.Module):
    def __init__(self, dim_context, emb_dim, first_stage_policy):
        super().__init__()
        self.first_stage_policy = first_stage_policy  

        self.fc = nn.Sequential(
            nn.Linear(dim_context, emb_dim),
            nn.ReLU()
        )

    def forward(self, x, A_k):
        context_repr = self.fc(x)  # transform context
        item_embs = self.first_stage_policy.item_embeddings(A_k)  # get top-k embeddings

        # compute inner product 
        scores = torch.matmul(item_embs, context_repr.unsqueeze(-1)).squeeze(-1)

        # compute softmax probabilities over top-k
        probs = torch.softmax(scores, dim=1)

        return probs