import pandas as pd


def load_ratings(path="data/ml-1m/ratings.dat"):
    return pd.read_csv(path, sep="::", engine="python",
                       names=["user", "item", "rating", "timestamp"])


def make_implicit(df, threshold=4):
    """Keep only ratings >= threshold as positive interactions."""
    df = df[df["rating"] >= threshold]
    return df[["user", "item", "timestamp"]].copy()


def remap_ids(df):
    """Turn user/item IDs into 0..N-1 so embeddings can use them."""
    user_map = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    item_map = {m: i for i, m in enumerate(sorted(df["item"].unique()))}
    df = df.copy()
    df["user"] = df["user"].map(user_map)
    df["item"] = df["item"].map(item_map)
    return df, len(user_map), len(item_map)


def leave_one_out_split(df):
    """Per user: latest interaction -> test, second latest -> validation, rest -> train."""
    df = df.sort_values(["user", "timestamp"])
    order = df.groupby("user").cumcount(ascending=False)  # 0 = latest
    test = df[order == 0]
    val = df[order == 1]
    train = df[order >= 2]
    return train, val, test


def prepare_data(path="data/ml-1m/ratings.dat", threshold=4):
    df = make_implicit(load_ratings(path), threshold)
    # drop users with fewer than 3 positives (cannot fill train/val/test)
    counts = df["user"].value_counts()
    df = df[df["user"].isin(counts[counts >= 3].index)]
    df, n_users, n_items = remap_ids(df)
    train, val, test = leave_one_out_split(df)
    return train, val, test, n_users, n_items


if __name__ == "__main__":
    train, val, test, n_users, n_items = prepare_data()
    print("users:", n_users, "items:", n_items)
    print("train:", len(train), "val:", len(val), "test:", len(test))