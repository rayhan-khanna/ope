"""Command line for one training run; run from the repository root with PYTHONPATH=. (see README)."""
import argparse

from benchmark.ope.opl_two_stage_train import train, DEFAULT_EPOCHS, NAIVE_CF_EPOCHS


def main():
    parser = argparse.ArgumentParser(description="Train one run of the two-stage retrieval grid.")
    parser.add_argument("--method", default="online_single_pref",
                        choices=("online_single_pref", "online_multi_pref", "naive_cf", "online_cf_eps"))
    parser.add_argument("--setting", default="latent", choices=("latent", "minrel"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=50, help="K: candidate-set size")
    parser.add_argument("--ranking_size", type=int, default=10, help="R: slate size")
    parser.add_argument("--n_pref", type=int, default=1, help="P: preference vectors per user (online_multi_pref)")
    parser.add_argument("--eval_k_list", type=lambda s: [int(k) for k in s.split(",")], default="20,50,100",
                        help="comma-separated deploy-time candidate sizes; sizes below --ranking_size are ignored")
    parser.add_argument("--credit", default=None, choices=("ca", "joint"),
                        help="ca: credit-assigned gradient, with learned allocation for online_multi_pref; joint: "
                             "plain gradient, round-robin heads; default ca in latent, joint in min-rel; CF ignores it")
    parser.add_argument("--minrel_lambda", type=float, default=5.5, help="diversity-bonus scale of the min-rel setting")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help=f"default {DEFAULT_EPOCHS}, or {NAIVE_CF_EPOCHS} for naive_cf")
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
