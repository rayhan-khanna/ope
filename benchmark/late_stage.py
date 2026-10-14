"""Greedy late stage of the min-rel setting and the slate values built on it."""
import torch


def greedy_slate(dataset, user_ids, candidates):
    """Pick L of the candidates greedily on each user's weighted relevance-plus-diversity objective."""
    rel_weight = (2.0 * dataset.user_w_rel[user_ids]).view(-1, 1)
    div_weight = (2.0 * dataset.user_w_div[user_ids]).view(-1, 1)
    L = dataset.L
    B, K = candidates.shape
    device = candidates.device
    div_coeff = dataset.minrel_lambda * L / (L * (L - 1) / 2.0)
    rel, dist = dataset.relevance_and_distance(user_ids, candidates)
    relu_rel = rel.clamp_min(0.0)
    selected = torch.zeros(B, K, dtype=torch.bool, device=device)
    chosen = torch.empty(B, L, dtype=torch.long, device=device)
    rel_total = torch.zeros(B, device=device)
    div_total = torch.zeros(B, device=device)
    div_gain = torch.zeros(B, K, device=device)
    rows = torch.arange(B, device=device)
    for step in range(L):
        score = (rel_weight * (rel_total.unsqueeze(1) + rel)
                 + div_weight * div_coeff * (div_total.unsqueeze(1) + div_gain))
        score[selected] = -1e9
        pick = score.argmax(dim=1)
        chosen[:, step] = pick
        selected[rows, pick] = True
        rel_total += rel[rows, pick]
        div_total += div_gain[rows, pick]
        div_gain = div_gain + torch.minimum(relu_rel, relu_rel[rows, pick].unsqueeze(1)) * dist[rows, pick, :]
    return candidates.gather(1, chosen)


def slate_value(dataset, user_ids, ranked):
    """Expected min-rel reward of each ranked slate."""
    return dataset.reward(user_ids, ranked).sum(1)


def minrel_value(dataset, user_ids, candidates):
    """Expected slate reward of a candidate set after the greedy late stage."""
    return slate_value(dataset, user_ids, greedy_slate(dataset, user_ids, candidates))


def slate_components(dataset, user_ids, ranked):
    """The two legs of a slate's min-rel reward: summed relevance and the scaled diversity term."""
    rel = (dataset.action_context[ranked] * dataset.user_context[user_ids].unsqueeze(1)).sum(-1).sum(-1)
    return rel, dataset.L * dataset.minrel_lambda * dataset.diversity(ranked, user_ids)
