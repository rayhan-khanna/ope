# Multi-preference retrieval for two-stage recommendation

Code for the experiments on learning the retrieval stage of a two-stage recommender with several preference
vectors per user. The retrieval policy proposes K candidates, a fixed late stage ranks R of them, and the policy
is trained online from slate rewards by policy gradient (credit-assigned in the latent setting). Two synthetic
settings are studied:

* `latent`: each interaction draws a hidden interest from the user's mixture over interest modes, so the best
  candidate set hedges over modes;
* `minrel`: the reward pays a relevance-gated diversity bonus, so the best slate mixes relevance and diversity.

## Layout

```
custom_obp/synthetic_bandit_dataset.py        synthetic users, items, logged data and rewards for both settings
custom_obp/policy.py                          the retrieval policy: preference vectors, item table, allocation, log-probabilities
custom_obp/cf_model.py                        the collaborative-filtering scorer used by the two baselines
benchmark/ope/run_two_stage.py                command line for one run
benchmark/ope/opl_two_stage_train.py          configuration, run names, checkpoints, dispatch
benchmark/ope/train_pg.py                     policy-gradient training loop, deployment and logging
benchmark/ope/train_cf.py                     offline and online CF baselines
benchmark/ope/latent_eval.py                  the latent setting's late stage, evaluation and reference values
benchmark/ope/late_stage.py                   the min-rel setting's greedy late stage
```

## Setup

```
pip install -r requirements.txt
wandb login
```

Every command below runs from the repository root with `PYTHONPATH=.`. Runs log to the wandb project
`two_stage_bandit` and save checkpoints under `checkpoints/<run name>/`.

## Training

The command line sets the method, the setting, the seed, the candidate budget K (`--top_k`), the number of
preference vectors P (`--n_pref`), the slate size R (`--ranking_size`), the estimator (`--credit`), the deploy-time
budgets (`--eval_k_list`, budgets below R are ignored), the min-rel bonus scale (`--minrel_lambda`, default 5.5) and
the training length (`--n_epochs`, default 100000, or 20000 for `naive_cf`); everything else is fixed in the code.

```
# latent setting, main arm: P=20 preference vectors with learned allocation, credit-assigned policy gradient
PYTHONPATH=. python benchmark/ope/run_two_stage.py --method online_multi_pref --top_k 50 --n_pref 20 --seed 0

# the single-preference policy with the same recipe
PYTHONPATH=. python benchmark/ope/run_two_stage.py --method online_single_pref --top_k 50 --seed 0

# min-rel setting (plain policy gradient, round-robin heads)
PYTHONPATH=. python benchmark/ope/run_two_stage.py --setting minrel --method online_multi_pref --top_k 50 --n_pref 20 --seed 0

# CF baselines: regression on logged data, and online epsilon-greedy CF
PYTHONPATH=. python benchmark/ope/run_two_stage.py --method naive_cf --top_k 50 --seed 0
PYTHONPATH=. python benchmark/ope/run_two_stage.py --method online_cf_eps --top_k 50 --seed 0
```

`--credit joint` trains a multi-preference policy with the plain estimator and round-robin heads (the credit
ablation).

## Logged metrics

Every key below also comes per user type with a `_type{0,1,2}` suffix: peaked, flat and mid users by the perplexity
of their interest mixture in the latent setting; relevance-heavy, diversity-heavy and balanced users in min-rel.

* `eval_batch_reward`: slate reward of the greedy deployment on all users, every epoch (every 50 for `naive_cf`);
* `relative_reward`: the same rescaled so that random retrieval is 0 and the oracle candidate set is 1 (per type for
  policy-gradient runs in the latent setting; absent for the CF baselines in min-rel);
* `val_reward_K{k}`: the deployed reward at each budget in `--eval_k_list`, every ten epochs (every 50 for
  `naive_cf`), with `relative_reward_K{k}` for the budgets other than K in the latent setting;
* `cand_*`: diversity of the deployed candidate sets (item clusters covered, plus the interest mass they cover in the
  latent setting or their pairwise dissimilarity in min-rel);
* `eval_batch_rel`, `eval_batch_div`: the relevance and diversity legs of the min-rel reward;
* `early_stage_top1_action_prob` (policy-gradient runs, per type `top1_prob_type{t}`): the mean probability of the
  first pick's most likely item on the training batches;
* `collapsed` (policy-gradient runs): 1 once the deployed reward falls below 65% of its best; training then stops
  and the last record is repeated for the remaining epochs so the seed still counts in group means.

Other keys: `batch_reward` (training or behaviour reward), `loss_policy` or `cf_mse`, and the references
`random_slate_reward` and `oracle_slate_reward`.

Every run saves `final.pt` and `latest.pt` at the end and `epoch<N>.pt` every 10000 epochs. Policy-gradient runs also
save `latest.pt` every epoch and `best.pt` on improvement until a collapse; `online_cf_eps` saves `latest.pt` every
1000 epochs.
