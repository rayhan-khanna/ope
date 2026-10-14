import torch
import torch.nn as nn


def gumbel_like(x):
    """Standard Gumbel noise shaped like x."""
    return -torch.log(-torch.log(torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)))


class MultiPreferencePolicy(nn.Module):
    """Retrieval policy: n_pref preference vectors per user scored against a learned item table."""

    def __init__(self, n_users, n_items, emb_dim, n_pref, logit_temp, single, alloc):
        super().__init__()
        self.n_pref, self.logit_temp, self.single, self.alloc = n_pref, logit_temp, single, alloc
        # two unused layers are initialised before the tables so seeds reproduce the reported runs
        nn.Linear(emb_dim, emb_dim)
        nn.Linear(emb_dim, emb_dim)
        self.item_embeddings = nn.Embedding(n_items, emb_dim)
        self.user_embeddings = nn.Embedding(n_users, n_pref * emb_dim)
        nn.init.uniform_(self.user_embeddings.weight, -1.0, 1.0)
        self.alloc_logits = nn.Embedding.from_pretrained(torch.zeros(n_users, n_pref), freeze=False)

    def head_vectors(self, user_ids):
        flat = self.user_embeddings(user_ids)
        return flat.view(flat.size(0), self.n_pref, -1)

    def alloc_probs(self, user_ids):
        return torch.softmax(self.alloc_logits(user_ids), dim=-1)

    def alloc_counts(self, user_ids, K):
        """Largest-remainder rounding of K * alloc_probs; rows sum to K."""
        pi = self.alloc_probs(user_ids)
        raw = pi * K
        base = raw.floor().long()
        short = K - base.sum(dim=1)
        frac_rank = raw.frac().argsort(dim=1, descending=True)
        bump = torch.zeros_like(base)
        ar = torch.arange(pi.size(1), device=pi.device).unsqueeze(0)
        bump.scatter_(1, frac_rank, (ar < short.unsqueeze(1)).long())
        return base + bump

    def retrieve(self, user_ids, K, greedy):
        """K distinct items per user, sampled or greedy; heads are chosen in turn or by the learned allocation."""
        device = user_ids.device
        E = self.item_embeddings.weight
        B, P = user_ids.size(0), self.n_pref
        heads = self.head_vectors(user_ids)
        if self.single:
            w = heads[:, 0, :]
            with torch.no_grad():
                logits = (w @ E.T) / self.logit_temp
                candidates = torch.topk(logits if greedy else logits + gumbel_like(logits), K, dim=1).indices
            return candidates, w[:, None, :].expand(B, K, -1)
        with torch.no_grad():
            if not self.alloc:
                head_of_pick = (torch.arange(K, device=device) % P).unsqueeze(0).expand(B, -1)
            elif greedy:
                sorted_counts, order = self.alloc_counts(user_ids, K).sort(dim=1, descending=True)
                picks = torch.arange(K, device=device).unsqueeze(0).expand(B, -1).contiguous()
                head_of_pick = order.gather(1, torch.searchsorted(sorted_counts.cumsum(dim=1), picks, right=True))
            else:
                head_of_pick = torch.multinomial(self.alloc_probs(user_ids), K, replacement=True)
        candidates = torch.empty(B, K, device=device, dtype=torch.long)
        taken = torch.zeros(B, E.size(0), device=device, dtype=torch.bool)
        rows = torch.arange(B, device=device)
        pick_vecs = []
        for k in range(K):
            w_k = heads[rows, head_of_pick[:, k], :]
            pick_vecs.append(w_k)
            with torch.no_grad():
                logits = ((w_k @ E.T) / self.logit_temp).masked_fill(taken, float("-inf"))
                pick = torch.argmax(logits if greedy else logits + gumbel_like(logits), dim=1)
            candidates[:, k] = pick
            taken[rows, pick] = True
        return candidates, torch.stack(pick_vecs, dim=1)

    def log_prob_candidates(self, candidates, pick_vecs):
        """Joint log-probability of the K picks, each a softmax of its own preference vector."""
        E = self.item_embeddings.weight
        logits = torch.einsum("bkd,nd->bkn", pick_vecs, E) / self.logit_temp
        chosen = logits.gather(2, candidates.unsqueeze(2)).squeeze(2)
        return (chosen - torch.logsumexp(logits, dim=2)).sum(dim=1)

    def log_inclusion_prob(self, user_ids, actions, K):
        """Credit-assigned log-probability that each ranked item is retrieved."""
        E = self.item_embeddings.weight
        head_vecs = self.head_vectors(user_ids)
        alloc_probs = self.alloc_probs(user_ids)
        P, L = self.n_pref, actions.shape[1]
        head_logits = torch.einsum("bpd,nd->bpn", head_vecs, E) / self.logit_temp
        with torch.no_grad():
            mixture_log_prob = torch.logsumexp(
                torch.log_softmax(head_logits, dim=-1) + alloc_probs.clamp_min(1e-12).log().unsqueeze(-1), dim=1)
            top = mixture_log_prob.topk(K + 1, dim=1).indices
            prefixes = torch.where(top[:, :K].unsqueeze(1) == actions.unsqueeze(-1),
                                   top[:, K].unsqueeze(1).unsqueeze(-1), top[:, :K].unsqueeze(1))
        head_log_norm = torch.logsumexp(head_logits, dim=-1)
        prefix_logits = head_logits.unsqueeze(2).expand(-1, -1, L, -1).gather(
            3, prefixes.unsqueeze(1).expand(-1, P, -1, -1))
        log_norms = [head_log_norm.unsqueeze(-1).expand(-1, -1, L).unsqueeze(-1)]
        running = prefix_logits[..., 0]
        for k in range(1, K):
            norm = head_log_norm.unsqueeze(-1)
            log_norms.append((norm + torch.log1p(-torch.exp((running - norm).clamp(max=-1e-6)))).unsqueeze(-1))
            running = torch.logaddexp(running, prefix_logits[..., k])
        normalizer = torch.cat(log_norms, dim=-1)
        action_logits = head_logits.gather(2, actions.unsqueeze(1).expand(-1, P, -1)).unsqueeze(-1)
        head_prob = torch.exp((action_logits - normalizer).clamp(max=-1e-6))
        inclusion = torch.einsum("bp,bplk->blk", alloc_probs, head_prob).clamp(max=1.0 - 1e-6)
        log_excluded = torch.sum(torch.log1p(-inclusion), dim=-1)
        return torch.log1p(-torch.exp(log_excluded.clamp(max=-1e-6)))
