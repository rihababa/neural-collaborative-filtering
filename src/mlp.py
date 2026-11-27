import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=(64, 32, 16)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        blocks = []
        in_dim = emb_dim * 2          # user + item concatenated
        for h in layers:
            blocks += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(in_dim, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        x = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids)], dim=-1)
        return self.out(self.mlp(x)).squeeze(-1)


if __name__ == "__main__":
    model = MLP(n_users=100, n_items=200)
    u = torch.randint(0, 100, (8,))
    i = torch.randint(0, 200, (8,))
    print(model(u, i).shape)   # should print torch.Size([8])