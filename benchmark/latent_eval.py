"""The latent setting's late stage, evaluation and reference values; every draw here has its own seeded generator."""
import torch

EVAL_SEED = 12345
EVAL_DRAWS = 20
SEARCH_DRAWS = 48
SEED_ORACLE = EVAL_SEED + 1
SEED_RANDOM = EVAL_SEED + 3
SEED_SEARCH = EVAL_SEED + 777
BUDGET_SEED_STRIDE = 7


def rank_under_z(dataset, user_ids, candidates, z):
    """The late stage under a realized interest: the top-L candidates by <[x; z], a>."""
    query = torch.cat([dataset.user_context[user_ids], z], dim=-1)
    rel = (dataset.action_context[candidates] * query.unsqueeze(1)).sum(-1)
    return candidates.gather(1, rel.topk(dataset.L, dim=1).indices)


def latent_value(dataset, candidates):
    """Expected slate reward of every user's candidate set, averaged over EVAL_DRAWS common interest draws."""
    g = torch.Generator().manual_seed(EVAL_SEED)
    user_ids = torch.arange(dataset.n_users, device=dataset.device)
    total = torch.zeros(dataset.n_users, device=dataset.device)
    for _ in range(EVAL_DRAWS):
        z = dataset.sample_z(user_ids, generator=g)
        total += dataset.reward(user_ids, rank_under_z(dataset, user_ids, candidates, z), z=z).sum(1)
    return total / EVAL_DRAWS


def random_candidates(dataset, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, dataset.n_items, (dataset.n_users, k), generator=g).to(dataset.device)


def relevance_draws(dataset, seed, n):
    """Relevance of every item to every user under n successive interest draws from a generator seeded with seed."""
    g = torch.Generator().manual_seed(seed)
    user_ids = torch.arange(dataset.n_users, device=dataset.device)
    base = dataset.user_context @ dataset.action_context[:, :dataset.dim].T
    item_z = dataset.action_context[:, dataset.dim:]
    for _ in range(n):
        yield base + dataset.sample_z(user_ids, generator=g) @ item_z.T


def sampled_oracle(dataset, k, seed):
    """k interest draws, each adding the best item not taken yet."""
    rows = torch.arange(dataset.n_users, device=dataset.device)
    taken = torch.zeros(dataset.n_users, dataset.n_items, dtype=torch.bool, device=dataset.device)
    picks = []
    for scores in relevance_draws(dataset, seed, k):
        scores[taken] = float("-inf")
        pick = scores.argmax(1)
        picks.append(pick)
        taken[rows, pick] = True
    return torch.stack(picks, dim=1)


def search_draws(dataset):
    """Relevance of every item under SEARCH_DRAWS fixed interest draws per user."""
    return torch.stack(list(relevance_draws(dataset, SEED_SEARCH, SEARCH_DRAWS)))


def greedy_set_oracle(rel_draws, k, L):
    """Greedy candidate set maximizing the mean over draws of the top-L relevance it contains."""
    U, N = rel_draws.shape[1:]
    rows = torch.arange(U, device=rel_draws.device)
    taken = torch.zeros(U, N, dtype=torch.bool, device=rel_draws.device)
    mean_rel = rel_draws.mean(0)
    chosen, picks = [], []
    for step in range(k):
        if step < L:
            gain = mean_rel
        else:
            tau = torch.stack(chosen, dim=-1).topk(L, dim=-1).values[..., -1]
            gain = (rel_draws - tau.unsqueeze(-1)).clamp_min(0).mean(0)
        pick = gain.masked_fill(taken, float("-inf")).argmax(1)
        picks.append(pick)
        taken[rows, pick] = True
        chosen.append(rel_draws[:, rows, pick])
    return torch.stack(picks, dim=1)


def anchor_values(dataset, k, rel_draws, random_seed, oracle_seed):
    """Per-user values at budget k of random candidates and of the two oracle sets, sampled and greedy.

    The oracle reference is the per-user better of the two oracle sets."""
    random_value = latent_value(dataset, random_candidates(dataset, k, random_seed))
    sampled = latent_value(dataset, sampled_oracle(dataset, k, oracle_seed))
    greedy = latent_value(dataset, greedy_set_oracle(rel_draws, k, dataset.L))
    return random_value, sampled, greedy


def budget_references(dataset, budgets, rel_draws):
    """Mean (random, oracle) values at each budget, drawn with that budget's own seeds."""
    refs = {}
    for k in budgets:
        stride = BUDGET_SEED_STRIDE * k
        random_value, sampled, greedy = anchor_values(dataset, k, rel_draws, EVAL_SEED + stride, SEED_ORACLE + stride)
        refs[k] = (random_value.mean().item(), torch.maximum(sampled, greedy).mean().item())
        print(f"[anchors] K'={k}: random={refs[k][0]:.2f} oracle={refs[k][1]:.2f}")
    return refs
