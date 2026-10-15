"""Command line for one training run; run from the repository root with PYTHONPATH=. (see README)."""
import argparse

from benchmark.rl_two_stage_train import train, DEFAULT_EPOCHS, NAIVE_CF_EPOCHS, DEFAULT_N_PREF


def main():
    parser = argparse.ArgumentParser(description="Train one run of the two-stage retrieval grid.")
    parser.add_argument("--method", default="online_single_pref",
                        choices=("online_single_pref", "online_multi_pref", "naive_cf", "online_cf_eps"))
    parser.add_argument("--setting", default="latent", choices=("latent", "minrel"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=50, help="K: candidate-set size")
    parser.add_argument("--ranking_size", type=int, default=10, help="L: ranking length chosen by the late stage")
    parser.add_argument("--n_pref", type=int, default=None,
                        help=f"P: preference vectors per user for online_multi_pref (default {DEFAULT_N_PREF})")
    parser.add_argument("--eval_k_list", type=lambda s: [int(k) for k in s.split(",")], default="20,50,100",
                        help="comma-separated deploy-time candidate sizes; sizes below --ranking_size are ignored")
    parser.add_argument("--credit", default=None, choices=("ca_swr", "ca", "joint"),
                        help="ca_swr: credit-assigned gradient under sampling with replacement, with learned "
                             "allocation for online_multi_pref (latent policy-gradient runs only, their default); ca: "
                             "the same without the replacement approximation; joint: vanilla gradient under sampling "
                             "with replacement, round-robin heads (the default otherwise)")
    parser.add_argument("--minrel_lambda", type=float, default=5.5, help="diversity-bonus scale of the min-rel setting")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help=f"default {DEFAULT_EPOCHS}, or {NAIVE_CF_EPOCHS} for naive_cf")
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
