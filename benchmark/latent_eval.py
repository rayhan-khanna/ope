"""The latent setting's late stage, evaluation and reference values; every draw here has its own seeded generator.

Shapes: U users, N items, B users in a batch, K candidates, L ranked items, S interest draws, DIM-entry interests.
"""
import torch

EVAL_SEED = 12345
EVAL_DRAWS = 20
SEARCH_DRAWS = 48
SEED_ORACLE = EVAL_SEED + 1
SEED_RANDOM = EVAL_SEED + 3
SEED_SEARCH = EVAL_SEED + 777
BUDGET_SEED_STRIDE = 7


def rank_under_z(dataset, user_ids, candidates, z):
    """The late stage under a realized interest: the top-L candidates by <[x; z], a>.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    user_ids: torch.Tensor, shape (B, )
        The users.

    candidates: torch.Tensor, shape (B, K)
        Each user's candidate set.

    z: torch.Tensor, shape (B, DIM)
        The realized interest of each interaction.

    Output
    ------
    ranked: torch.Tensor, shape (B, L)
        The L most relevant candidates, in order.

    """
    query = torch.cat([dataset.user_context[user_ids], z], dim=-1)
    rel = (dataset.action_context[candidates] * query.unsqueeze(1)).sum(-1)
    return candidates.gather(1, rel.topk(dataset.L, dim=1).indices)


def latent_value(dataset, candidates):
    """Expected slate reward of every user's candidate set, averaged over EVAL_DRAWS common interest draws.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    candidates: torch.Tensor, shape (U, M)
        M candidates for every user.

    Output
    ------
    value: torch.Tensor, shape (U, )
        The reward of the ranking the late stage builds from each candidate set, averaged over the draws.

    """
    g = torch.Generator().manual_seed(EVAL_SEED)
    user_ids = torch.arange(dataset.n_users, device=dataset.device)
    total = torch.zeros(dataset.n_users, device=dataset.device)
    for _ in range(EVAL_DRAWS):
        z = dataset.sample_z(user_ids, generator=g)
        total += dataset.reward(user_ids, rank_under_z(dataset, user_ids, candidates, z), z=z).sum(1)
    return total / EVAL_DRAWS


def random_candidates(dataset, k, seed):
    """k uniformly random items for every user, drawn with replacement.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    k: int
        The number of items per user.

    seed: int
        The seed of the draw.

    Output
    ------
    candidates: torch.Tensor, shape (U, k)
        The random items.

    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, dataset.n_items, (dataset.n_users, k), generator=g).to(dataset.device)


def relevance_draws(dataset, seed, n):
    """Relevance of every item to every user under n successive interest draws.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    seed: int
        The seed of the interest draws.

    n: int
        The number of draws.

    Output
    ------
    rel: generator of torch.Tensor, each of shape (U, N)
        n tensors, one per draw: <x, a^x> + <z, a^z> for every user and item, with one interest z per user.

    """
    g = torch.Generator().manual_seed(seed)
    user_ids = torch.arange(dataset.n_users, device=dataset.device)
    base = dataset.user_context @ dataset.action_context[:, :dataset.dim].T
    item_z = dataset.action_context[:, dataset.dim:]
    for _ in range(n):
        yield base + dataset.sample_z(user_ids, generator=g) @ item_z.T


def sampled_oracle(dataset, k, seed):
    """k interest draws, each adding the best item not taken yet.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    k: int
        The candidate-set size.

    seed: int
        The seed of the interest draws.

    Output
    ------
    candidates: torch.Tensor, shape (U, k)
        The oracle candidate set of every user.

    """
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
    """Relevance of every item under SEARCH_DRAWS fixed interest draws per user.

    These are the scenarios the greedy set oracle optimizes; references and policies are scored on separate draws.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    Output
    ------
    rel_draws: torch.Tensor, shape (SEARCH_DRAWS, U, N)
        The relevance matrices of relevance_draws with seed SEED_SEARCH, stacked.

    """
    return torch.stack(list(relevance_draws(dataset, SEED_SEARCH, SEARCH_DRAWS)))


def greedy_set_oracle(rel_draws, k, L):
    """Greedy candidate set maximizing the mean over draws of the top-L relevance it contains.

    The first L picks take the highest mean relevance; each later pick adds the item with the largest expected gain
    E[max(0, rel − τ)], where τ is the current L-th best relevance in the set under each draw.

    Input
    ------
    rel_draws: torch.Tensor, shape (S, U, N)
        The relevance of every item to every user under each draw.

    k: int
        The candidate-set size.

    L: int
        The ranking length.

    Output
    ------
    candidates: torch.Tensor, shape (U, k)
        The oracle candidate set of every user.

    """
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

    The oracle reference is the per-user better of the two oracle sets.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    k: int
        The candidate-set size.

    rel_draws: torch.Tensor, shape (S, U, N)
        The search draws of the greedy set oracle.

    random_seed: int
        The seed of the random candidates.

    oracle_seed: int
        The seed of the sampled oracle's interest draws.

    Output
    ------
    random_value: torch.Tensor, shape (U, )
        The value of random candidates.

    sampled: torch.Tensor, shape (U, )
        The value of the sampled oracle set.

    greedy: torch.Tensor, shape (U, )
        The value of the greedy set oracle.

    """
    random_value = latent_value(dataset, random_candidates(dataset, k, random_seed))
    sampled = latent_value(dataset, sampled_oracle(dataset, k, oracle_seed))
    greedy = latent_value(dataset, greedy_set_oracle(rel_draws, k, dataset.L))
    return random_value, sampled, greedy


def budget_references(dataset, budgets, rel_draws):
    """Mean (random, oracle) values at each budget; random candidates and the sampled oracle are seeded per budget.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    budgets: list of int
        The deploy-time candidate-set sizes.

    rel_draws: torch.Tensor, shape (S, U, N)
        The search draws of the greedy set oracle.

    Output
    ------
    refs: dict of int to (float, float)
        The mean random and oracle values at each budget.

    """
    refs = {}
    for k in budgets:
        stride = BUDGET_SEED_STRIDE * k
        random_value, sampled, greedy = anchor_values(dataset, k, rel_draws, EVAL_SEED + stride, SEED_ORACLE + stride)
        refs[k] = (random_value.mean().item(), torch.maximum(sampled, greedy).mean().item())
        print(f"[anchors] K'={k}: random={refs[k][0]:.2f} oracle={refs[k][1]:.2f}")
    return refs
