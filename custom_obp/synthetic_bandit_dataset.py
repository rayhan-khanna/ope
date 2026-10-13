import numpy as np
import torch
import torch.nn.functional as F

N_ITEMS = 1000
N_USERS = 1000
N_LOGGED = 100_000
DIM = 5
N_MODES = 5
ITEM_STD = 0.3
Z_SIGMA = 0.3
X_STD = 0.15
MAX_MODE_COS = 0.5
MODE_THETA_SCALE = 1.7
TEMP_LO, TEMP_HI, TEMP_GAIN = 0.25, 4.0, 2.0
PERPLEXITY_BINS = (1.6, 2.6)
HET_DELTA = 0.1
HET_THETA_SCALE = 2.0
LOGGING_TEMP = 1.1
REWARD_STD = 1.0


def regular_simplex(n, dim):
    """n unit vectors in R^dim that sum to zero with pairwise cosine -1/(n-1)."""
    V = np.eye(n) - 1.0 / n
    basis = np.linalg.svd(V)[2][:n - 1].T
    out = np.zeros((n, dim))
    out[:, :n - 1] = V @ basis
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class SyntheticBanditDataset:
    """Synthetic users and items; `latent` draws a hidden interest per interaction, `minrel` pays a diversity bonus."""

    def __init__(self, setting, K, seed, R, minrel_lambda, device):
        self.latent = setting == "latent"
        self.K, self.R = K, R
        self.minrel_lambda = minrel_lambda
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.n_items, self.dim = N_ITEMS, DIM
        self.theta = torch.tensor(self.rng.uniform(-1.0, 1.0, (DIM, 3)).astype("float32"),
                                  device=device) * HET_THETA_SCALE
        if self.latent:
            scale = np.linalg.norm(self.separated_centers(), axis=1).mean()
            centers = (regular_simplex(N_MODES, DIM) * scale).astype("float32")
            assignments = self.rng.randint(0, N_MODES, size=N_ITEMS)
            item_z = centers[assignments] + self.rng.normal(scale=ITEM_STD, size=(N_ITEMS, DIM)).astype("float32")
            item_x = self.rng.normal(scale=X_STD, size=(N_ITEMS, DIM)).astype("float32")
            self.action_context = torch.tensor(np.concatenate([item_x, item_z], axis=1), device=device)
            self.mode_centers = torch.tensor(centers, device=device)
            self.theta_z = torch.tensor(self.rng.normal(size=(DIM, N_MODES)).astype("float32"),
                                        device=device) * MODE_THETA_SCALE
            self.theta_temp = torch.tensor(self.rng.normal(size=(DIM,)).astype("float32"), device=device) / np.sqrt(DIM)
        else:
            centers = self.rng.normal(size=(N_MODES, DIM)).astype("float32")
            assignments = self.rng.randint(0, N_MODES, size=N_ITEMS)
            items = centers[assignments] + self.rng.normal(scale=ITEM_STD, size=(N_ITEMS, DIM)).astype("float32")
            self.action_context = torch.tensor(items, device=device)
        self.item_cluster_assignments = torch.tensor(assignments, device=device, dtype=torch.long)

    def separated_centers(self):
        """Mode directions rejection-sampled until every pair has cosine <= MAX_MODE_COS; only their scale is used."""
        while True:
            c = self.rng.normal(size=(N_MODES, DIM)).astype("float32")
            unit = c / np.linalg.norm(c, axis=1, keepdims=True)
            cos = unit @ unit.T
            np.fill_diagonal(cos, -1.0)
            if cos.max() <= MAX_MODE_COS:
                return c

    def sample_context(self, n):
        return torch.tensor(self.rng.uniform(low=-1.0, high=1.0, size=(n, DIM)),
                            dtype=torch.float32, device=self.device)

    def sample_users(self, n):
        return torch.randint(0, self.n_users, (n,), device=self.device)

    def assign_minrel_users(self):
        probs = torch.softmax(self.user_context @ self.theta, dim=-1)
        self.user_types = torch.multinomial(probs, num_samples=1).squeeze(-1)
        delta = torch.tensor(self.rng.uniform(low=0.0, high=HET_DELTA, size=self.n_users).astype("float32"),
                             device=self.device)
        self.user_w_rel = torch.where(self.user_types == 0, 1.0 - delta,
                                      torch.where(self.user_types == 1, delta, 0.5 + delta))
        self.user_w_div = 1.0 - self.user_w_rel
        torch.randperm(self.n_users)  # kept so seeds reproduce the reported runs (it fed a diagnostic print)
        counts = torch.bincount(self.user_types, minlength=3).tolist()
        print(f"[dataset] minrel: user types rel-heavy/div-heavy/balanced = {counts[0]}/{counts[1]}/{counts[2]}")

    def assign_latent_users(self):
        x = self.user_context
        u01 = torch.sigmoid((x @ self.theta_temp) * TEMP_GAIN)
        temp = TEMP_LO * (TEMP_HI / TEMP_LO) ** u01
        self.user_z_weights = w = torch.softmax((x @ self.theta_z) / temp.unsqueeze(1), dim=1)
        perplexity = torch.exp(-(w * torch.log(w.clamp_min(1e-12))).sum(1))
        lo, hi = PERPLEXITY_BINS
        self.user_types = torch.full((x.shape[0],), 2, device=self.device)
        self.user_types[perplexity < lo] = 0
        self.user_types[perplexity > hi] = 1
        counts = torch.bincount(self.user_types, minlength=3).tolist()
        print(f"[dataset] latent: mode perplexity mean {perplexity.mean():.2f}; "
              f"users peaked/flat/mid = {counts[0]}/{counts[1]}/{counts[2]}")

    def sample_z(self, user_ids, generator=None):
        """One interest draw per user; a CPU generator makes evaluation draws identical everywhere."""
        draw_device = self.device if generator is None else "cpu"
        w = self.user_z_weights[user_ids].to(draw_device)
        mode = torch.multinomial(w, 1, generator=generator).squeeze(1).to(self.device)
        eps = torch.randn(w.shape[0], DIM, generator=generator, device=draw_device).to(self.device)
        return self.mode_centers[mode] + Z_SIGMA * eps

    def relevance_and_distance(self, user_ids, items):
        """Relevance <x, a> of each user's items, and the pairwise cosine distance between those items."""
        emb = self.action_context[items]
        unit = F.normalize(emb, dim=-1)
        rel = (self.user_context[user_ids].unsqueeze(1) * emb).sum(-1)
        return rel, 1.0 - unit @ unit.transpose(1, 2)

    def diversity(self, slates, user_ids):
        """Relevance-gated mean pairwise dissimilarity of each slate, the min-rel bonus term."""
        R = slates.shape[1]
        rel, dist = self.relevance_and_distance(user_ids, slates)
        rel = rel.clamp_min(0.0)
        gate = torch.minimum(rel.unsqueeze(2), rel.unsqueeze(1))
        upper = torch.triu(torch.ones(R, R, dtype=torch.bool, device=slates.device), diagonal=1)
        return (gate * dist * upper).sum(dim=(1, 2)) / (R * (R - 1) / 2)

    def reward(self, user_ids, ranked, noisy=False, z=None):
        """Per-position reward of each user's ranked slate."""
        R = ranked.shape[1]
        idx = user_ids.repeat_interleave(R)
        items = self.action_context[ranked.reshape(-1)]
        users = self.user_context[idx]
        if self.latent:
            expected = torch.sum(torch.cat([users, z.repeat_interleave(R, dim=0)], dim=-1) * items, dim=1)
        else:
            relevance = torch.sum(users * items, dim=1)
            div = self.diversity(ranked, user_ids).repeat_interleave(R)
            expected = 2.0 * self.user_w_rel[idx] * relevance + 2.0 * self.user_w_div[idx] * (self.minrel_lambda * div)
        if noisy:
            expected = expected + torch.randn_like(expected) * REWARD_STD
        return expected.view(-1, R)

    def sample_logging_action(self, context_row):
        probs = torch.softmax((context_row.unsqueeze(0) @ self.action_context.T) / LOGGING_TEMP, dim=-1)[0]
        return torch.multinomial(probs, 1).item()

    def logged_feedback(self, single_stage):
        """Create the users and the softmax-policy log naive_cf trains on: slates (latent) or single items (min-rel)."""
        self.n_users = N_USERS
        self.user_context = self.sample_context(N_USERS)
        if self.latent:
            self.assign_latent_users()
        else:
            self.assign_minrel_users()
        user_ids = torch.arange(N_USERS, device=self.device).repeat(N_LOGGED // N_USERS + 1)[:N_LOGGED]
        user_ids = user_ids[torch.randperm(N_LOGGED)]
        context = self.user_context[user_ids]
        if self.latent:
            zbar = self.user_z_weights[user_ids] @ self.mode_centers
            scores = (torch.cat([context, zbar], dim=1) @ self.action_context.T) / LOGGING_TEMP
            g1 = -torch.log(-torch.log(torch.rand_like(scores)))
            candidates = torch.topk(scores + g1, self.K, dim=1).indices
            cs = scores.gather(1, candidates)
            g2 = -torch.log(-torch.log(torch.rand_like(cs)))
            ranked = candidates.gather(1, (cs + g2).topk(self.R, dim=1).indices)
            rewards = self.reward(user_ids, ranked, noisy=True, z=self.sample_z(user_ids))
            return {"context": context, "action": ranked, "reward": rewards}
        if single_stage:
            actions = torch.tensor([self.sample_logging_action(row) for row in context]).to(self.device)
            torch.randn(N_LOGGED, device=self.device)  # reward noise of the reported runs' log, kept for their seeds
            return {"context": context, "action": actions.unsqueeze(1)}
        # the min-rel slate log is never read; its draws are replayed so seeds reproduce the reported runs
        torch.rand(N_LOGGED, self.n_items, device=self.device)
        for _ in range(N_LOGGED):
            torch.empty(self.K, device=self.device).exponential_()
        for _ in range(N_LOGGED):
            torch.randn(self.R, device=self.device)
