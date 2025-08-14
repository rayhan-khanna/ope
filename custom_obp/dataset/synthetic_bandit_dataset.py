import torch
import torch.nn as nn
from sklearn.utils import check_random_state
from custom_obp.policy.action_policies import SoftmaxPolicy

class CustomSyntheticBanditDataset:
    """Generates synthetic bandit feedback dataset."""
    def __init__(self, 
                 n_actions=10000, 
                 dim_context=10, 
                 top_k=10, 
                 ranking_size=5,
                 reward_std=1.0, 
                 random_state=42,
                 action_policy=None, 
                 device="cpu", 
                 single_stage=True,
                 second_stage_policy=None): 
        
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.top_k = top_k
        self.ranking_size = ranking_size
        self.reward_std = reward_std
        self.device = torch.device(device) 
        
        self.random_ = check_random_state(random_state)
        
        self.action_context = torch.tensor(
            self.random_.normal(size=(n_actions, dim_context)), dtype=torch.float32
        ).to(self.device)

        self.single_stage = single_stage

        if action_policy is None:
            self.action_policy = SoftmaxPolicy(dim_context)
        else:
            self.action_policy = action_policy

        self.second_stage_policy = second_stage_policy

    def sample_context(self, n):
        return torch.tensor(
            self.random_.normal(size=(n, self.dim_context)),
            dtype=torch.float32,
            device=self.device
        )

    def sample_policy_full(self, context):
        return torch.tensor([
            self.action_policy.sample_action(context[i], self.action_context)
            for i in range(context.shape[0])
        ], dtype=torch.long).to(self.device)

    def generate_data(self, context, user_ids):
        if self.single_stage:
            actions = self.sample_policy_full(context).unsqueeze(1)
            rewards = self.reward_function(user_ids, actions.squeeze(1)).unsqueeze(1)
            return {"context": context, "user_id": user_ids, "action": actions, "reward": rewards}

        candidates = self.candidate_selection(context)
        ranked_actions, ranked_probs = self.rank_from_candidates(context, candidates)
        rewards = torch.stack([
            self.reward_function(user_ids[i].repeat(self.ranking_size), ranked_actions[i])
            for i in range(len(user_ids))
        ])

        return {
            "context": context,
            "user_id": user_ids,
            "candidates": candidates,
            "action": ranked_actions,
            "action_probs": ranked_probs,
            "reward": rewards
        }

    def rank_from_candidates(self, context, candidates):
        """Sample a ranking of size `ranking_size` from candidates using logging policy."""
        ranked_actions = []
        ranked_probs = []

        for i in range(context.size(0)):
            cand = candidates[i]
            # softmax probs over candidates
            probs = self.action_policy.probs(context[i], self.action_context[cand])
            chosen = torch.multinomial(probs, self.ranking_size, replacement=False)  # sample ranking
            ranked_actions.append(cand[chosen])
            ranked_probs.append(probs[chosen])

        return torch.stack(ranked_actions), torch.stack(ranked_probs)

    class PrefNet(nn.Module):
        def __init__(self, dim_ctx: int, dim_emb: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim_ctx + dim_emb, hidden), nn.ReLU(),
                nn.Linear(hidden, 2 * dim_emb)
            )

        def forward(self, ctx: torch.Tensor, agg: torch.Tensor):
            h = torch.cat([ctx, agg], dim=-1)
            mu, log_sigma = self.net(h).chunk(2, dim=-1)
            return mu, log_sigma.exp()

    def sample_k_prefs(
        self,
        fb,
        n_pref_per_user: int,
        pref_net,
        gamma: float = 1.0,
        tau_pref: float = 0.1,
    ):
        ctx_all, row_emb_all = fb["context"], fb["row_emb"]
        a_all, r_all, p_all = fb["action"], fb["reward"], fb["pscore"]
        u_all = fb["user_id"]
        device, D = ctx_all.device, row_emb_all.size(-1)

        uniq_u, inv = torch.unique(u_all, return_inverse=True)
        B = len(uniq_u)

        if n_pref_per_user == 1:
            first_row = torch.stack([(inv == i).nonzero(as_tuple=True)[0][0]
                                    for i in range(B)], dim=0)
            return {
                "context": ctx_all[first_row],
                "action" : a_all [first_row].unsqueeze(1),
                "reward" : r_all [first_row].unsqueeze(1),
                "pscore" : p_all [first_row].unsqueeze(1),
                "user_id": uniq_u,
            }

        rows_left = [(inv == k).nonzero(as_tuple=True)[0].to(device) for k in range(B)]
        agg_emb = torch.zeros(B, D, device=device) 
        chosen_idx = [[] for _ in range(B)] 

        for j in range(n_pref_per_user):
            len_rows = torch.tensor([len(r) for r in rows_left], device=device)
            active = len_rows > 0
            if not active.any(): break

            act_id = active.nonzero(as_tuple=True)[0]
            first_rows = torch.stack([
                (inv == k).nonzero(as_tuple=True)[0][0]
                for k in act_id
            ])
            ctx_batch = ctx_all[first_rows] 
            agg_batch = agg_emb[act_id] / (j if j else 1)

            mu, sigma = pref_net(ctx_batch, agg_batch)
            w_batch = mu + tau_pref * sigma * torch.randn_like(mu)

            logits_split = []
            ptr = 0
            for k in act_id:
                rows = rows_left[k]
                emb = row_emb_all[rows]
                logits_split.append((emb @ w_batch[ptr].T) / gamma)
                ptr += 1
            logits = torch.cat(logits_split)

            gumbel = -torch.empty_like(logits).exponential_().log()
            logits = (logits + gumbel).split(len_rows[act_id].tolist())

            for idx_u, k in enumerate(act_id):
                if logits[idx_u].numel() == 0: continue
                loc = logits[idx_u].argmax().item()
                chosen_idx[k].append(rows_left[k][loc])

                agg_emb[k] += row_emb_all[rows_left[k][loc]]
                keep = torch.arange(len_rows[k], device=device) != loc
                rows_left[k] = rows_left[k][keep]

        mask_users = torch.tensor([len(c) > 0 for c in chosen_idx], device=device)
        sel_rows = [torch.stack(chosen_idx[k]) for k in mask_users.nonzero(as_tuple=True)[0]]
        sel_rows = torch.stack(sel_rows)

        return {
            "context": ctx_all[sel_rows[:, 0]],
            "action" : a_all [sel_rows],
            "reward" : r_all [sel_rows],
            "pscore" : p_all [sel_rows],
            "user_id": uniq_u[mask_users],
        }

    def candidate_selection(self, context):
        scores = torch.matmul(context, self.action_context.T)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores)))
        noisy_scores = scores + gumbel_noise
        top_k_indices = torch.topk(noisy_scores, self.top_k, dim=1).indices
        return top_k_indices

    def sample_policy(self, candidates, context):
        return torch.tensor([
            self.action_policy.select_action(candidates[i], context[i], self.action_context)
            for i in range(candidates.shape[0])
        ], dtype=torch.long).to(self.device)

    def vendi_score(self, embeddings: torch.Tensor):
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        K = embeddings @ embeddings.T
        diag = torch.sqrt(torch.diag(K)).unsqueeze(0)
        K = K / (diag.T @ diag + 1e-12)  # cosine normalization
        eps = 1e-6 * torch.eye(K.size(0), device=K.device)  # jitter for stability
        eigvals = torch.linalg.eigvalsh(K / K.size(0) + eps)
        eigvals = eigvals.clamp(min=1e-12)
        entropy = -(eigvals * torch.log(eigvals)).sum()
        return torch.exp(entropy)

    def reward_function(self, user_ids, actions):
        if self.second_stage_policy is not None:
            item_vecs = self.second_stage_policy.first_stage_policy.item_embeddings(actions)
        else:
            item_vecs = self.action_context[actions]

        user_vecs = self.user_embeddings[user_ids.long()]
        relevance = torch.sum(user_vecs * item_vecs, dim=1) # relevance per item
        noise = torch.randn_like(relevance) * self.reward_std

        if self.ranking_size > 1:  
            # compute diversity per ranked list (batch)
            batch_size = user_ids.size(0) // self.ranking_size
            diversity_scores = []
            for i in range(batch_size):
                block = item_vecs[i * self.ranking_size:(i + 1) * self.ranking_size]
                diversity_scores.append(self.vendi_score(block))
            diversity_scores = torch.stack(diversity_scores)

            # repeat diversity score across ranking positions
            diversity_scores = diversity_scores.repeat_interleave(self.ranking_size)
            gamma = 1.2  # hyperparam to weight diversity
            return relevance + gamma * diversity_scores + noise
        else:
            # single-action case: no diversity term
            return relevance + noise

    def obtain_batch_bandit_feedback(self, n_samples, n_users):
        self.n_users = n_users
        self.user_embeddings = torch.tensor(
            self.random_.normal(size=(n_users, self.dim_context)),
            dtype=torch.float32,
            device=self.device
        )

        user_ids = torch.arange(n_users, device=self.device).repeat(n_samples // n_users + 1)[:n_samples]
        user_ids = user_ids[torch.randperm(n_samples)]
        context = self.user_embeddings[user_ids]

        data = self.generate_data(context, user_ids)

        if self.single_stage:
            pi_b_full = torch.stack([
                self.action_policy.probs(context[i], self.action_context)
                for i in range(n_samples)
            ]).to(self.device)
            pscore = pi_b_full[torch.arange(n_samples), data["action"].squeeze()]
            return {"n_samples": n_samples, **data, "pscore": pscore}

        # multi-stage: compute joint prob of ranking
        log_pscores = (data["action_probs"].log()).sum(dim=1)
        return {"n_samples": n_samples, **data, "pscore": log_pscores.exp()}