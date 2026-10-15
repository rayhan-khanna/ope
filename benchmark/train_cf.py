"""The CF baselines, offline regression and online epsilon-greedy; in min-rel they regress item relevance.

Shapes: B users in a batch, N items, DIM context entries.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from env.synthetic_bandit_dataset import REWARD_STD
from env.cf_model import NaiveCF
from benchmark.latent_eval import search_draws, budget_references
from benchmark.train_pg import (BATCH_SIZE, STEPS_PER_EPOCH, SWEEP_INTERVAL, SNAPSHOT_INTERVAL, References,
                                deployment, deployment_log, serve_slates)

CF_LR = 1e-2
CF_EPS = 0.2
NAIVE_CF_EVAL_INTERVAL = 50
EPS_PRINT_INTERVAL = 100
EPS_SAVE_INTERVAL = 1000


def cf_references(run):
    """Random and oracle values at the trained and every other deploy budget, latent setting only."""
    dataset = run.dataset
    refs = budget_references(dataset, sorted(set(run.eval_k_list) | {dataset.K}), search_draws(dataset))
    random_ref, oracle_ref = refs.pop(dataset.K)
    return References(random_ref, oracle_ref, {}, refs, {})


def build_cf(run):
    dataset = run.dataset
    model = NaiveCF(dataset.dim, dataset.n_items, dataset.action_context.shape[1]).to(dataset.device)
    return model, optim.Adagrad(model.parameters(), lr=CF_LR)


def cf_scores(model, context):
    """Scores of every item for each context.

    Input
    ------
    model: NaiveCF
        The CF scorer.

    context: torch.Tensor, shape (B, DIM)
        The user contexts.

    Output
    ------
    scores: torch.Tensor, shape (B, N)
        <W x + b, e_a> for every item a.

    """
    return model.project_context(context) @ model.item_embedding.weight.T


@torch.no_grad()
def cf_log(run, model, refs, budgets):
    """wandb fields of the CF model's greedy deployment at the trained budget and at each budget in budgets."""
    dataset = run.dataset
    scores = cf_scores(model, dataset.user_context)
    deploy = deployment(dataset, scores.topk(dataset.K, dim=1).indices)
    sweep = {k: deploy.reward if k == dataset.K else deployment(dataset, scores.topk(k, dim=1).indices).reward
             for k in budgets}
    return deployment_log(dataset, refs, deploy, sweep)


def legs_text(dataset, log):
    return "" if dataset.latent else f" | Rel: {log['eval_batch_rel']:.4f} | Div: {log['eval_batch_div']:.4f}"


def train_naive_cf(run, feedback):
    dataset = run.dataset
    refs = cf_references(run) if dataset.latent else None
    model, optimizer = build_cf(run)
    actions = feedback["action"]
    rows = actions.size(1)  # 1 for single-stage logged data, L for rankings
    context_flat = feedback["context"].repeat_interleave(rows, dim=0)
    actions_flat = actions.reshape(-1)
    if dataset.latent:
        target = feedback["reward"].reshape(-1)
    else:
        target = ((context_flat * dataset.action_context[actions_flat]).sum(-1)
                  + REWARD_STD * torch.randn_like(actions_flat, dtype=torch.float32))

    for epoch in range(run.n_epochs):
        optimizer.zero_grad()
        loss = F.mse_loss(model(context_flat, actions_flat), target)
        loss.backward()
        optimizer.step()
        if epoch % NAIVE_CF_EVAL_INTERVAL == 0:
            log = {"epoch": epoch, "cf_mse": loss.item(), **cf_log(run, model, refs, run.eval_k_list)}
            wandb.log(log)
            print(f"[Epoch {epoch}] naive_cf | MSE: {loss.item():.4f} | EvalReward: {log['eval_batch_reward']:.4f}"
                  + legs_text(dataset, log))
        if (epoch + 1) % SNAPSHOT_INTERVAL == 0:
            run.save(run.bundle("naive_cf", model, epoch + 1), f"epoch{epoch + 1}.pt")
    log = {"epoch": run.n_epochs, "cf_mse": loss.item(), **cf_log(run, model, refs, run.eval_k_list)}
    wandb.log(log)
    print(f"[naive_cf FINAL] EvalReward: {log['eval_batch_reward']:.4f}" + legs_text(dataset, log))
    run.save(run.bundle("naive_cf", model, run.n_epochs), "final.pt", "latest.pt")


def train_online_cf_eps(run):
    dataset = run.dataset
    K, L = dataset.K, dataset.L
    refs = cf_references(run) if dataset.latent else None
    n_random = round(CF_EPS * K)
    model, optimizer = build_cf(run)

    for epoch in range(1, run.n_epochs + 1):
        behavior_rewards, losses = [], []
        for _ in range(STEPS_PER_EPOCH):
            user_ids = dataset.sample_users(BATCH_SIZE)
            context = dataset.user_context[user_ids]
            with torch.no_grad():
                random_items = torch.rand(BATCH_SIZE, dataset.n_items, device=dataset.device).topk(n_random).indices
                scores = cf_scores(model, context).scatter(1, random_items, float("-inf"))
                candidates = torch.cat([random_items, scores.topk(K - n_random, dim=1).indices], dim=1)
            ranked, rewards = serve_slates(dataset, user_ids, candidates)
            context_flat = context.repeat_interleave(L, dim=0)
            if dataset.latent:
                target = rewards.reshape(-1)
            else:
                ranked_flat = ranked.reshape(-1)
                target = ((context_flat * dataset.action_context[ranked_flat]).sum(-1)
                          + REWARD_STD * torch.randn_like(ranked_flat, dtype=torch.float32))
            behavior_rewards.append(rewards.sum(1).mean().item())
            loss = F.mse_loss(model(context_flat, ranked.reshape(-1)), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        budgets = run.eval_k_list if epoch % SWEEP_INTERVAL == 0 else []
        log = {"epoch": epoch, "cf_mse": np.mean(losses), "batch_reward": np.mean(behavior_rewards),
               **cf_log(run, model, refs, budgets)}
        wandb.log(log)
        if epoch % EPS_PRINT_INTERVAL == 0:
            print(f"[Epoch {epoch}] online_cf_eps ε={CF_EPS:g} | MSE {log['cf_mse']:.4f} "
                  f"| behavior {log['batch_reward']:.3f} | greedy-deploy {log['eval_batch_reward']:.3f}")
        if epoch % EPS_SAVE_INTERVAL == 0:
            bundle = run.bundle("naive_cf", model, epoch)
            run.save(bundle, "latest.pt")
            if epoch % SNAPSHOT_INTERVAL == 0:
                run.save(bundle, f"epoch{epoch}.pt")
    run.save(run.bundle("naive_cf", model, run.n_epochs), "final.pt", "latest.pt")
