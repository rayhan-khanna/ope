# Intrinsic and Extrinsic Diversity-aware Candidate Retrieval in Two-stage Decisions

This repository contains the code for the synthetic experiments of the paper "Intrinsic and Extrinsic
Diversity-aware Candidate Retrieval in Two-stage Decisions".

## Overview

In a two-stage decision system, an early-stage ranker (ESR) retrieves a candidate set of K items and a late-stage
ranker (LSR) chooses a ranking of length L from those candidates. Here the ESR is learned online by policy gradient
from the reward of the final ranking, while the LSR stays fixed. The ESR can represent each user with several
preference vectors, so a single candidate set can cover several directions of interest.

Two synthetic settings are studied:

* `latent` (extrinsic diversity): each interaction draws a latent interest z from the user's mixture over interest
  modes, and an item's relevance depends on both the user context x and z. The LSR observes z and ranks the
  candidates by their relevance under it, but the ESR never observes z: it scores items with preference vectors
  learned for each user. Because of this information gap, the ESR cannot know which interest will be active, so its
  candidate set needs diversity for risk-hedging: covering several modes guards against missing the one that turns
  out to matter.
* `minrel` (intrinsic diversity): the reward itself pays a relevance-gated diversity bonus, so the best ranking mixes
  relevant and mutually dissimilar items. The LSR builds it greedily.

In the latent setting the ESR is trained with the credit-assigned policy gradient (CA-PG) of
[facebookresearch/early_stage_retrieval](https://github.com/facebookresearch/early_stage_retrieval). CA-PG credits
each ranked item's probability of entering the candidate set with that item's own reward, instead of crediting the
joint probability of the whole set with the total reward as vanilla policy gradient does. With CA-PG, an ESR with
several preference vectors also learns how to split the K slots among them. The min-rel reward is not additive per
item, so that setting uses vanilla policy gradient, with the preference vectors filling the slots in turn.

Two collaborative-filtering (CF) baselines complete the comparison: `naive_cf` regresses logged data offline, and
`online_cf_eps` learns online, filling 20% of the K slots with random items and the rest with its top-scored items.
In the latent setting they regress the observed reward of each ranked item; in min-rel they regress item relevance,
so they cannot learn the diversity bonus.

## Layout

```
env/synthetic_bandit_dataset.py      synthetic users, items, logged data and rewards for both settings
env/policy.py                        the ESR: preference vectors, item table, allocation, log-probabilities
env/cf_model.py                      the CF scorer used by the two baselines
benchmark/run_two_stage.py           command line for one run
benchmark/rl_two_stage_train.py      configuration, run names, checkpoints, dispatch
benchmark/train_pg.py                policy-gradient training loop, deployment and logging
benchmark/train_cf.py                offline and online CF baselines
benchmark/latent_eval.py             the latent setting's LSR, evaluation and reference values
benchmark/late_stage.py              the min-rel setting's greedy LSR
```

## Setup

Use Python 3.10, as the reported runs did; the pinned versions install on Python 3.9 to 3.12.

```
pip install -r requirements.txt
wandb login
```

Every command below runs from the repository root with `PYTHONPATH=.`. Runs log to the wandb project
`two_stage_bandit` and save checkpoints under `checkpoints/<run name>/`.

## Training

The command line sets the method, the setting (`--setting latent` or `minrel`), the seed, the candidate budget K
(`--top_k`), the number of preference vectors P (`--n_pref`, default 5 for `online_multi_pref`), the ranking length L
(`--ranking_size`), the estimator (`--credit ca` or `joint`; `ca` exists only for the policy-gradient methods in the
latent setting, where it is the default), the deploy-time budgets (`--eval_k_list`, budgets below L are ignored), the
min-rel bonus scale (`--minrel_lambda`, default 5.5) and the training length (`--n_epochs`, default 100000, or 20000
for `naive_cf`); everything else is fixed in the code. The commands below use the training lengths of the reported
runs, which used seeds 0 to 9.

```
# latent setting, main arm: P=20 preference vectors, CA-PG with learned slot allocation (100000 epochs)
PYTHONPATH=. python benchmark/run_two_stage.py --method online_multi_pref --top_k 50 --n_pref 20 --seed 0

# the single-preference ESR with the same recipe
PYTHONPATH=. python benchmark/run_two_stage.py --method online_single_pref --top_k 50 --seed 0 --n_epochs 600000

# the credit ablation: vanilla policy gradient, preference vectors filling the slots in turn
PYTHONPATH=. python benchmark/run_two_stage.py --method online_multi_pref --top_k 50 --n_pref 20 --credit joint \
    --seed 0 --n_epochs 500000
PYTHONPATH=. python benchmark/run_two_stage.py --method online_single_pref --top_k 50 --credit joint --seed 0 \
    --n_epochs 700000

# min-rel setting (vanilla policy gradient)
PYTHONPATH=. python benchmark/run_two_stage.py --setting minrel --method online_multi_pref --top_k 50 --n_pref 20 \
    --seed 0 --n_epochs 400000
PYTHONPATH=. python benchmark/run_two_stage.py --setting minrel --method online_single_pref --top_k 50 --seed 0 \
    --n_epochs 400000

# CF baselines at their default lengths (20000 and 100000 epochs); add --setting minrel for the min-rel setting
PYTHONPATH=. python benchmark/run_two_stage.py --method naive_cf --top_k 50 --seed 0
PYTHONPATH=. python benchmark/run_two_stage.py --method online_cf_eps --top_k 50 --seed 0
```

Run names do not include the training length, so runs of one arm at different lengths share a wandb group.

## Logged metrics

Unless noted, each key below also comes per user type with a `_type{0,1,2}` suffix: peaked, flat and mid users by the
perplexity of their interest mixture in the latent setting; relevance-heavy, diversity-heavy and balanced users in
min-rel.

* `eval_batch_reward`: reward of the ranking after greedy retrieval for all users, every epoch (every 50 for
  `naive_cf`);
* `relative_reward`: the same rescaled so that random retrieval is 0 and the oracle candidate set is 1 (per type for
  policy-gradient runs in the latent setting; absent for the CF baselines in min-rel);
* `val_reward_K{k}`: the reward at each budget in `--eval_k_list`, every ten epochs (every 50 for `naive_cf`), with
  `relative_reward_K{k}`, overall only, for the budgets other than K in the latent setting;
* `cand_*`: diversity of the retrieved candidate sets (item clusters covered, plus the interest mass they cover in the
  latent setting or their pairwise dissimilarity in min-rel);
* `eval_batch_rel`, `eval_batch_div`: the relevance and diversity terms of the min-rel reward;
* `early_stage_top1_action_prob` (policy-gradient runs, per type `top1_prob_type{t}`): the mean probability of the
  first pick's most likely item on the training batches;
* `collapsed` (policy-gradient runs, overall only): 1 once the reward falls below 65% of its best; training then
  stops and the last record is repeated for the remaining epochs so the seed still counts in group means.

Other keys, overall only: `batch_reward` (training or behaviour reward), `loss_policy` or `cf_mse`, and the references
`random_slate_reward` and `oracle_slate_reward` (absent for the CF baselines in min-rel). Policy-gradient runs in the
latent setting also log `oracle_sampled_reward` and `oracle_greedy_reward`, the two oracle constructions whose
per-user maximum is the oracle reference. At the trained K, policy-gradient and CF runs draw their references with
different seeds, so their `relative_reward` scales differ slightly; compare the two on `eval_batch_reward`.

## Checkpoints

Every run saves `final.pt` and `latest.pt` at the end and `epoch<N>.pt` every 10000 epochs. Policy-gradient runs also
save `latest.pt` every epoch and `best.pt` on improvement until a collapse; `online_cf_eps` saves `latest.pt` every
1000 epochs. The metadata of `final.pt` records the epoch at which training stopped. Run names keep the tag layout of
the reported runs, for example `R10` for L=10 and `v6` or `v5` for the latent or min-rel setting.

## Reproducibility

The reported runs used the pinned versions on Linux x86 machines, most of them with NVIDIA RTX A6000 GPUs. On another
platform the results agree in distribution but not seed by seed: the latent setting builds its interest modes from a
singular value decomposition whose basis depends on the linear-algebra library, and PyTorch's random draws differ
between GPU models, between GPU and CPU, and between CPU builds.
