import numpy as np
import pandas as pd
import torch


def build_eval_set(eval_df, exclude_dfs, n_items, n_neg=99, seed=42):
    """For each user: [positive item] + n_neg random items the user never interacted with.
    Positive is always at index 0. Call once, reuse for every model."""
    rng = np.random.default_rng(seed)
    all_df = pd.concat([eval_df] + list(exclude_dfs))
    seen = all_df.groupby("user")["item"].apply(set).to_dict()

    users, cands = [], []
    for u, pos in zip(eval_df["user"], eval_df["item"]):
        negs = set()
        while len(negs) < n_neg:
            j = int(rng.integers(n_items))
            if j not in seen[u]:
                negs.add(j)
        users.append(u)
        cands.append([pos] + sorted(negs))
    return np.array(users), np.array(cands)


@torch.no_grad()
def evaluate(model, users, cands, k=10, device="cpu"):
    """Returns (Recall@K, NDCG@K). Works with any model: model(user_ids, item_ids) -> scores."""
    model.eval()
    n, c = cands.shape
    u = torch.as_tensor(users).repeat_interleave(c).to(device)
    i = torch.as_tensor(cands).reshape(-1).to(device)
    scores = model(u, i).reshape(n, c)

    # rank of the positive = how many negatives score >= it (0 = best)
    rank = (scores[:, 1:] >= scores[:, :1]).sum(dim=1)
    hit = (rank < k).float()
    recall = hit.mean().item()
    ndcg = (hit / torch.log2(rank.float() + 2)).mean().item()
    return recall, ndcg


if __name__ == "__main__":
    from preprocessing import prepare_data
    from mlp import MLP

    train, val, test, n_users, n_items = prepare_data()
    users, cands = build_eval_set(val, [train], n_items)
    model = MLP(n_users, n_items)          # untrained model
    r, n = evaluate(model, users, cands, k=10)
    print(f"Untrained MLP  Recall@10={r:.4f}  NDCG@10={n:.4f}")