"""Policy-gradient training of the retrieval policy, plus the deployment and logging the CF baselines share.

Shapes: U users, B users in a batch, K candidates, L ranked items.
"""
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from env.synthetic_bandit_dataset import N_MODES
from env.policy import MultiPreferencePolicy
from benchmark.late_stage import greedy_slate, minrel_value, slate_value, slate_components
from benchmark.latent_eval import (latent_value, rank_under_z, search_draws, anchor_values, budget_references,
                                   SEED_RANDOM, SEED_ORACLE, EVAL_DRAWS)

ITEM_LR, PREF_LR = 0.01, 0.03
# credit-assigned estimators: ca_swr under sampling with replacement (the default), ca without it (the earlier runs)
CA_CREDITS = ("ca_swr", "ca")
BATCH_SIZE = 128
STEPS_PER_EPOCH = 10
SWEEP_INTERVAL = 10
SNAPSHOT_INTERVAL = 10000
COLLAPSE_FRAC = 0.65
REWARD_SCALE_BETA = 0.99
VAL_DRAW_USERS = 10000
MINREL_ORACLE_USERS = 256

# reference values: scalars at the trained budget, per user type and per other deploy budget, plus constant
# fields logged with them (the latent oracle's two constructions)
References = namedtuple("References", "random oracle by_type by_budget components")
# a greedy deployment: per-user reward, the (rel, div) legs (None in the latent setting) and the candidate sets
Deploy = namedtuple("Deploy", "reward rel div candidates")


def log_per_type(log, name, values, types):
    for t in range(3):
        mask = types == t
        log[f"{name}_type{t}"] = values[mask].mean().item() if mask.any() else float("nan")


def relative(value, random, oracle):
    """Position of a value between the random (0) and oracle (1) references."""
    return (value - random) / max(oracle - random, 1e-9)


def candidate_diversity(dataset, candidates):
    """Diversity of every user's candidate set: clusters covered, and interest mass or pairwise dissimilarity.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    candidates: torch.Tensor, shape (U, K)
        The candidate set of every user.

    Output
    ------
    diversity: dict of str to torch.Tensor, each of shape (U, )
        "eff_clusters" and "n_clusters", the effective and the plain number of item clusters covered; latent:
        "mass_covered" and "mass_full", the interest mass of the modes with at least one and at least L candidates;
        min-rel: "pair_div" and "pair_div_gated", the mean pairwise dissimilarity, plain and relevance-gated.

    """
    K = candidates.shape[1]
    counts = F.one_hot(dataset.item_cluster_assignments[candidates], N_MODES).sum(1).float()
    share = counts / K
    out = {"eff_clusters": torch.exp(-(share * torch.log(share.clamp_min(1e-12))).sum(1)),
           "n_clusters": (counts > 0).sum(1).float()}
    if dataset.latent:
        w = dataset.user_z_weights
        out["mass_covered"] = (w * (counts > 0)).sum(1)
        out["mass_full"] = (w * (counts >= dataset.L)).sum(1)
        return out
    rel, dist = dataset.relevance_and_distance(torch.arange(dataset.n_users, device=dataset.device), candidates)
    rel = rel.clamp_min(0)
    pair = dist * ~torch.eye(K, dtype=torch.bool, device=dist.device)
    out["pair_div"] = pair.sum((1, 2)) / (K * (K - 1))
    out["pair_div_gated"] = (torch.minimum(rel.unsqueeze(2), rel.unsqueeze(1)) * pair).sum((1, 2)) / (K * (K - 1))
    return out


def deployment(dataset, candidates):
    """Value of every user's candidate set under the fixed late stage.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    candidates: torch.Tensor, shape (U, K)
        The candidate set of every user.

    Output
    ------
    deploy: Deploy
        "reward" (U, ), the expected reward of each user's ranking; "rel" and "div" (U, ), the min-rel reward's two
        legs (None in the latent setting); "candidates" (U, K), the input.

    """
    if dataset.latent:
        return Deploy(latent_value(dataset, candidates), None, None, candidates)
    user_ids = torch.arange(dataset.n_users, device=dataset.device)
    ranked = greedy_slate(dataset, user_ids, candidates)
    rel, div = slate_components(dataset, user_ids, ranked)
    return Deploy(slate_value(dataset, user_ids, ranked), rel, div, candidates)


def deployment_log(dataset, refs, deploy, sweep):
    """wandb fields of a deployment and of the rewards at the swept budgets, overall and per user type."""
    types = dataset.user_types
    log = {"eval_batch_reward": deploy.reward.mean().item()}
    log_per_type(log, "eval_batch_reward", deploy.reward, types)
    if refs is not None:
        log["random_slate_reward"], log["oracle_slate_reward"] = refs.random, refs.oracle
        log.update(refs.components)
        log["relative_reward"] = relative(log["eval_batch_reward"], refs.random, refs.oracle)
        for t, (random_t, oracle_t) in refs.by_type.items():
            log[f"relative_reward_type{t}"] = relative(log[f"eval_batch_reward_type{t}"], random_t, oracle_t)
    if deploy.rel is not None:
        log["eval_batch_rel"], log["eval_batch_div"] = deploy.rel.mean().item(), deploy.div.mean().item()
        log_per_type(log, "eval_batch_rel", deploy.rel, types)
        log_per_type(log, "eval_batch_div", deploy.div, types)
    for name, values in candidate_diversity(dataset, deploy.candidates).items():
        log[f"cand_{name}"] = values.mean().item()
        log_per_type(log, f"cand_{name}", values, types)
    for k, reward_k in sweep.items():
        log[f"val_reward_K{k}"] = reward_k.mean().item()
        if refs is not None and k in refs.by_budget:
            log[f"relative_reward_K{k}"] = relative(log[f"val_reward_K{k}"], *refs.by_budget[k])
        log_per_type(log, f"val_reward_K{k}", reward_k, types)
    return log


def serve_slates(dataset, user_ids, candidates):
    """The late stage ranks each user's candidates and the user returns noisy per-position rewards.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    user_ids: torch.Tensor, shape (B, )
        The users.

    candidates: torch.Tensor, shape (B, K)
        Each user's candidate set.

    Output
    ------
    ranked: torch.Tensor, shape (B, L)
        The ranking the late stage builds; in the latent setting under a freshly drawn interest.

    rewards: torch.Tensor, shape (B, L)
        The observed reward of each position, with noise.

    """
    if dataset.latent:
        z = dataset.sample_z(user_ids)
        ranked = rank_under_z(dataset, user_ids, candidates, z)
        return ranked, dataset.reward(user_ids, ranked, noisy=True, z=z)
    ranked = greedy_slate(dataset, user_ids, candidates)
    return ranked, dataset.reward(user_ids, ranked, noisy=True)


class RewardScale:
    """Running per-type scale of the slate reward, so every user type carries equal gradient weight."""

    def __init__(self, device):
        self.scale = torch.ones(3, device=device)
        self.seen = torch.zeros(3, dtype=torch.bool, device=device)

    def update(self, slate_reward, types):
        """Update each type's running RMS with a batch and return the batch's scales.

        Input
        ------
        slate_reward: torch.Tensor, shape (B, )
            The observed reward of each user's ranking.

        types: torch.Tensor, shape (B, )
            The type of each user.

        Output
        ------
        scale: torch.Tensor, shape (B, )
            Each user's type scale divided by the mean over the types seen so far; rewards are divided by it.

        """
        for t in range(3):
            mask = types == t
            if mask.any():
                rms = slate_reward[mask].pow(2).mean().clamp_min(1e-12).sqrt()
                self.scale[t] = (rms if not self.seen[t]
                                 else REWARD_SCALE_BETA * self.scale[t] + (1.0 - REWARD_SCALE_BETA) * rms)
                self.seen[t] = True
        target = self.scale[self.seen].mean()
        return (self.scale[types] / target).clamp_min(1e-6)


def build_policy(run):
    dataset = run.dataset
    emb_dim = dataset.action_context.shape[1]
    policy = MultiPreferencePolicy(dataset.n_users, dataset.n_items, emb_dim, run.P, logit_temp=run.logit_temp,
                                   single=run.method == "online_single_pref", alloc=run.alloc).to(dataset.device)
    nn.init.uniform_(policy.item_embeddings.weight, -1.0, 1.0)
    nn.Linear(dataset.dim, emb_dim)  # the retired learned late stage's init draw; seeds reproduce the reported runs
    item_table = policy.item_embeddings.weight
    optimizer = optim.Adagrad([
        {"params": [item_table], "lr": ITEM_LR},
        {"params": [p for p in policy.parameters() if p is not item_table], "lr": PREF_LR},
    ])
    return policy, optimizer


def latent_references(run):
    """Random and oracle values on every user, overall, per type and per deploy budget."""
    dataset = run.dataset
    K = dataset.K
    rel_draws = search_draws(dataset)
    random_value, sampled, greedy = anchor_values(dataset, K, rel_draws, SEED_RANDOM, SEED_ORACLE)
    oracle_value = torch.maximum(sampled, greedy)
    random_ref, oracle_ref = random_value.mean().item(), oracle_value.mean().item()
    components = {"oracle_sampled_reward": sampled.mean().item(), "oracle_greedy_reward": greedy.mean().item()}
    print(f"[anchors] K={K}: random={random_ref:.4f} oracle={oracle_ref:.4f} "
          f"(sampled={components['oracle_sampled_reward']:.4f} greedy={components['oracle_greedy_reward']:.4f}; "
          f"{dataset.n_users} users, {EVAL_DRAWS} interest draws)")
    by_type = {}
    for t in range(3):
        mask = dataset.user_types == t
        if mask.any():
            by_type[t] = (random_value[mask].mean().item(), oracle_value[mask].mean().item())
    by_budget = budget_references(dataset, [k for k in run.eval_k_list if k != K], rel_draws)
    return References(random_ref, oracle_ref, by_type, by_budget, components)


def minrel_references(dataset, random_users, random_candidates, oracle_users):
    """Random value of uniform candidate sets; oracle value of the greedy slate over all items."""
    random_ref = minrel_value(dataset, random_users, random_candidates).mean().item()
    oracle_users = oracle_users[:MINREL_ORACLE_USERS]
    all_items = torch.arange(dataset.n_items, device=dataset.device).unsqueeze(0).expand(len(oracle_users), -1)
    oracle_ref = minrel_value(dataset, oracle_users, all_items).mean().item()
    print(f"[anchors] random {random_ref:.4f} ({len(random_users)} users) "
          f"oracle {oracle_ref:.4f} (greedy on all items, {len(oracle_users)} users)")
    return References(random_ref, oracle_ref, {}, {}, {})


@torch.no_grad()
def evaluate(dataset, policy, k):
    """Greedy deployment of the policy at budget k on every user.

    Input
    ------
    dataset: SyntheticBanditDataset
        The environment.

    policy: MultiPreferencePolicy
        The retrieval policy.

    k: int
        The deploy-time candidate-set size.

    Output
    ------
    deploy: Deploy
        As returned by deployment, for the policy's greedy candidate sets of size k.

    """
    user_ids = torch.arange(dataset.n_users, device=dataset.device)
    return deployment(dataset, policy.retrieve(user_ids, k, greedy=True)[0])


def train_step(run, policy, optimizer, reward_scale):
    """One policy-gradient update on a fresh batch of users.

    CA-PG minimises −Σ_l (r_l / scale) · log p(a_l), with log p from log_pick_prob (ca_swr) or log_inclusion_prob
    (ca). Vanilla policy gradient minimises −(Σ_l r_l / scale) · log P(candidates), with log P from
    log_prob_candidates.

    Input
    ------
    run: Run
        The run's dataset and configuration.

    policy: MultiPreferencePolicy
        The retrieval policy.

    optimizer: torch.optim.Optimizer
        The optimizer of the policy.

    reward_scale: RewardScale
        The running per-type reward scale.

    Output
    ------
    stats: dict
        "loss" (float); "slate_rewards", the B observed ranking rewards (list of float); "top1_prob" (B, ), the
        probability of the first pick's most likely item; "batch_types" (B, ), the users' types.

    """
    dataset = run.dataset
    user_ids = dataset.sample_users(BATCH_SIZE)
    candidates, pick_vecs = policy.retrieve(user_ids, dataset.K, greedy=False)
    ranked, position_rewards = serve_slates(dataset, user_ids, candidates)
    slate_reward = position_rewards.sum(dim=1)
    batch_types = dataset.user_types[user_ids]
    scale = reward_scale.update(slate_reward, batch_types)
    if run.credit in CA_CREDITS:
        log_incl = (policy.log_pick_prob(user_ids, ranked) if run.credit == "ca_swr"
                    else policy.log_inclusion_prob(user_ids, ranked, dataset.K))
        loss = -((position_rewards / scale.unsqueeze(1)) * log_incl).sum(1).mean()
    else:
        log_prob = policy.log_prob_candidates(candidates, pick_vecs)
        loss = -((slate_reward / scale) * log_prob).mean()
    with torch.no_grad():  # how peaked the first pick's preference vector is over the items
        top1_prob = torch.softmax((pick_vecs[:, 0, :] @ policy.item_embeddings.weight.T) / policy.logit_temp,
                                  dim=-1).max(dim=1).values
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "slate_rewards": slate_reward.tolist(), "top1_prob": top1_prob,
            "batch_types": batch_types}


def epoch_log(dataset, refs, steps, deploy, sweep, epoch):
    """The wandb record of one policy-gradient epoch."""
    log = {
        "epoch": epoch,
        "loss_policy": np.mean([s["loss"] for s in steps]),
        "batch_reward": np.mean([r for s in steps for r in s["slate_rewards"]]),
        "early_stage_top1_action_prob": np.mean([s["top1_prob"].mean().item() for s in steps]),
    }
    log_per_type(log, "top1_prob", torch.cat([s["top1_prob"] for s in steps]),
                 torch.cat([s["batch_types"] for s in steps]))
    log.update(deployment_log(dataset, refs, deploy, sweep))
    return log


def train_policy_gradient(run):
    dataset = run.dataset
    K = dataset.K
    policy, optimizer = build_policy(run)
    # the three reference draws are unused in the latent setting but keep the training stream of the reported runs
    random_users = dataset.sample_users(dataset.n_users)
    random_candidates = torch.randint(0, dataset.n_items, (dataset.n_users, K), device=dataset.device)
    oracle_users = dataset.sample_users(dataset.n_users)
    refs = (latent_references(run) if dataset.latent
            else minrel_references(dataset, random_users, random_candidates, oracle_users))
    dataset.sample_users(VAL_DRAW_USERS)  # the old validation batch draw, kept so seeds reproduce the reported runs

    reward_scale = RewardScale(dataset.device)
    best_reward = float("-inf")
    frozen_log, last_sweep = None, {}
    trained_epochs = 0
    for epoch in range(1, run.n_epochs + 1):
        sweep_epoch = epoch % SWEEP_INTERVAL == 0
        if frozen_log is not None:  # after a collapse the last record is repeated so the seed still counts
            log = dict(frozen_log, epoch=epoch)
            if sweep_epoch:
                log.update(last_sweep)
            wandb.log(log)
            continue
        steps = [train_step(run, policy, optimizer, reward_scale) for _ in range(STEPS_PER_EPOCH)]
        trained_epochs = epoch
        bundle = run.bundle("first_stage", policy, epoch)
        run.save(bundle, "latest.pt")
        if epoch % SNAPSHOT_INTERVAL == 0:
            run.save(bundle, f"epoch{epoch}.pt")
        deploy = evaluate(dataset, policy, K)
        sweep = ({k: deploy.reward if k == K else evaluate(dataset, policy, k).reward for k in run.eval_k_list}
                 if sweep_epoch else {})
        log = epoch_log(dataset, refs, steps, deploy, sweep, epoch)
        eval_reward = log["eval_batch_reward"]
        collapsed = best_reward > 0 and eval_reward < COLLAPSE_FRAC * best_reward
        log["collapsed"] = float(collapsed)
        if sweep:
            last_sweep = {k: v for k, v in log.items() if k.startswith("val_")}
        wandb.log(log)
        print(f"[Epoch {epoch}] Method: {run.method} | Loss: {log['loss_policy']:.4f} "
              f"| AvgReward(batch): {log['batch_reward']:.4f} | Top1: {log['early_stage_top1_action_prob']:.4f}")
        if eval_reward > best_reward:
            best_reward = eval_reward
            run.save(bundle, "best.pt")
            print(f"  -> new best eval reward {best_reward:.4f}, saved best.pt")
        elif collapsed:
            print(f"  !! collapse: eval reward {eval_reward:.4f} < {COLLAPSE_FRAC:.0%} of best {best_reward:.4f}; "
                  f"training stops and these metrics are carried forward so the seed still counts")
            frozen_log = {k: v for k, v in log.items() if not k.startswith("val_")}
    run.save(run.bundle("first_stage", policy, trained_epochs), "final.pt", "latest.pt")
