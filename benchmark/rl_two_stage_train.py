"""Train one run: a policy-gradient retrieval policy or a CF baseline, in the latent or min-rel setting.

Shapes: U users, N items, DIM context entries.
"""
import os
from dataclasses import dataclass
from datetime import datetime

import torch
import wandb

from env.synthetic_bandit_dataset import (SyntheticBanditDataset, N_ITEMS, N_LOGGED, N_MODES, ITEM_STD, X_STD,
                                          MAX_MODE_COS, HET_DELTA, HET_THETA_SCALE, LOGGING_TEMP, REWARD_STD)
from benchmark.train_cf import train_naive_cf, train_online_cf_eps, CF_EPS
from benchmark.train_pg import train_policy_gradient, ITEM_LR, PREF_LR, CA_CREDITS

PG_METHODS = ("online_single_pref", "online_multi_pref")
DEFAULT_EPOCHS = 100_000
NAIVE_CF_EPOCHS = 20_000
DEFAULT_N_PREF = 5
LOGIT_TEMP = {"latent": 1.0, "minrel": 0.5}
WANDB_PROJECT = "diversity-aware-candidate-retrieval"
WANDB_GROUP_MAX = 127
RUN_TAG = f"het_d{HET_DELTA:g}_hsingle_unified_mog{N_MODES}s{ITEM_STD:g}"


def emb_names(pg):
    """Embedding mode and its run-name tag: per-user preference tables for the policy, a context projection for CF."""
    return ("multi_user_emb", "muemb") if pg else ("learned_user_learned_item", "lrulri")


def run_names(method, latent, credit, alloc, K, L, P, seed, logit_temp, minrel_lambda, cf_eps, stamp):
    """wandb run and group names; the tag layout keys the logged experiments and must not change, so the
    ranking length L is still written as R."""
    pg = method in PG_METHODS
    tag = f"_{RUN_TAG}_{'v6' if latent else 'v5'}_ns{REWARD_STD:g}_rse"
    if credit in CA_CREDITS:
        tag += "_caswr" if credit == "ca_swr" else "_ca"
    if alloc:
        tag += "_alloc"
    if pg:
        tag += "_rn"
    tag += f"_lz_geo{MAX_MODE_COS:g}-{X_STD:g}-{ITEM_STD:g}_sx" if latent else "_mr"
    if pg and logit_temp != 1.0:
        tag += f"_lt{logit_temp:g}"
    if method == "online_cf_eps":
        tag += f"_eps{cf_eps:g}_trel"
    reward_str = f"rmodehet_d{HET_DELTA:g}_ts{HET_THETA_SCALE:g}"
    emb_tag = emb_names(pg)[1]
    lr_tag = f"_lr{ITEM_LR:g}-{PREF_LR:g}" if pg else ""
    name = (f"{method}{tag}_rows{N_LOGGED}_A{N_ITEMS}_K{K}_R{L}_T{LOGGING_TEMP:g}_P{P}"
            f"_g{minrel_lambda:g}_seed{seed}_{reward_str}_emb{emb_tag}{lr_tag}_{stamp}")
    group_lambda = "" if latent else f"_g{minrel_lambda:g}"
    group = f"{method}{tag}_K{K}_R{L}_P{P}{group_lambda}_{reward_str}_emb{emb_tag}{lr_tag}"[:WANDB_GROUP_MAX]
    return name, group


@dataclass
class Run:
    """What a training path needs: the dataset, the resolved configuration and where to save."""
    dataset: SyntheticBanditDataset
    method: str
    credit: str
    alloc: bool
    P: int
    logit_temp: float
    n_epochs: int
    eval_k_list: list
    run_dir: str
    meta: dict
    cf_eps: float = CF_EPS

    def bundle(self, key, module, epoch):
        """Checkpoint contents: the trained model and the environment's ground truth.

        Input
        ------
        key: str
            The checkpoint key of the model: "first_stage" for the policy, "naive_cf" for the CF scorer.

        module: torch.nn.Module
            The trained model.

        epoch: int
            The last trained epoch.

        Output
        ------
        bundle: dict
            key: the model's state_dict; "action_context" (N, 2·DIM) latent or (N, DIM) min-rel, the true item vectors;
            "user_embeddings" (U, DIM), the true user contexts x; "user_types" (U, ), each user's type 0, 1 or 2 (see
            SyntheticBanditDataset); "meta", the wandb config plus the epoch; min-rel only: "Theta" (DIM, 3), the
            user-type parameters, and "user_w_rel", "user_w_div" (U, ). The latent interest model (mode_centers,
            user_z_weights) is not saved.

        """
        dataset = self.dataset
        out = {key: module.state_dict(), "action_context": dataset.action_context,
               "user_embeddings": dataset.user_context, "user_types": dataset.user_types,
               "meta": dict(self.meta, epoch=epoch)}
        if not dataset.latent:
            out["Theta"] = dataset.theta
            out["user_w_rel"], out["user_w_div"] = dataset.user_w_rel, dataset.user_w_div
        return out

    def save(self, bundle, *names):
        for name in names:
            path = os.path.join(self.run_dir, name)
            tmp = path + f".tmp.{os.getpid()}"
            torch.save(bundle, tmp)
            os.replace(tmp, path)


def train(method, seed, setting, credit, top_k, ranking_size, n_pref, eval_k_list, minrel_lambda, n_epochs,
          cf_eps=CF_EPS):
    """Resolve one run's configuration, build its dataset and train it."""
    latent = setting == "latent"
    pg = method in PG_METHODS
    if credit is None:
        credit = "ca_swr" if latent and pg else "joint"
    if credit in CA_CREDITS and not (latent and pg):
        raise ValueError(f"--credit {credit} is a latent-setting policy-gradient estimator: the min-rel reward is not "
                         "additive per item, and the CF baselines take no estimator")
    alloc = method == "online_multi_pref" and credit in CA_CREDITS
    logit_temp = LOGIT_TEMP[setting] if pg else 1.0
    minrel_lambda = 0.0 if latent else minrel_lambda
    if n_epochs is None:
        n_epochs = NAIVE_CF_EPOCHS if method == "naive_cf" else DEFAULT_EPOCHS
    eval_k_list = [k for k in eval_k_list if k >= ranking_size]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticBanditDataset(setting, K=top_k, seed=seed, L=ranking_size, minrel_lambda=minrel_lambda,
                                     device=device)
    feedback = dataset.logged_feedback(single_stage=(method == "naive_cf"))
    K, L = top_k, ranking_size
    if method == "online_multi_pref":
        P = DEFAULT_N_PREF if n_pref is None else n_pref
    else:
        P = 1 if method == "online_single_pref" else 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name, group = run_names(method, latent, credit, alloc, K, L, P, seed, logit_temp, minrel_lambda, cf_eps, stamp)
    run_dir = os.path.join("checkpoints", name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[setup] {method} setting={setting} K={K} L={L} P={P} credit={credit} alloc={alloc} seed={seed} "
          f"n_epochs={n_epochs} -> {run_dir}")
    # the reported runs' wandb config, without the keys of removed features, so new runs line up with them
    emb_mode, emb_tag = emb_names(pg)
    meta = {"method": method, "seed": seed, "n_epochs": n_epochs, "rows": N_LOGGED, "n_actions": N_ITEMS,
            "top_k": K, "temperature": LOGGING_TEMP, "P": P, "emb_dim": dataset.action_context.shape[1],
            "run_tag": RUN_TAG, "credit": credit, "alloc": alloc, "emb_mode": emb_mode, "emb_tag": emb_tag,
            "reward_mode": "heterogeneous", "het_delta": HET_DELTA, "het_theta_scale": HET_THETA_SCALE,
            "cf_eps": cf_eps, "reward_noise_std": REWARD_STD, "minrel_reward": not latent,
            "minrel_lambda": minrel_lambda, "logit_temp": logit_temp, "minrel_relu": not latent, "run_dir": run_dir}
    if pg:
        meta.update(item_lr_learned=ITEM_LR, pref_lr_learned=PREF_LR)
    wandb.init(project=WANDB_PROJECT, name=name, group=group, settings=wandb.Settings(init_timeout=300), config=meta)
    run = Run(dataset=dataset, method=method, credit=credit, alloc=alloc, P=P, logit_temp=logit_temp,
              n_epochs=n_epochs, eval_k_list=eval_k_list, run_dir=run_dir, meta=meta, cf_eps=cf_eps)
    if method == "naive_cf":
        train_naive_cf(run, feedback)
    elif method == "online_cf_eps":
        train_online_cf_eps(run)
    else:
        train_policy_gradient(run)
    wandb.finish()
