import pandas as pd
from scipy.sparse import dok_matrix


def to_map(xs):
    map_to = dict(enumerate(xs))
    map_from = {v: k for k, v in map_to.items()}
    return map_from, map_to


# Movielens ML-20M is expected to be extracted in ./ml-20m/
def load_movielens_ratings():
    ratings = pd.read_csv("ml-20m/ratings.csv")

    # Map users/items into a more dense structure, as
    # not all movies have ratings
    user_to_id, id_to_user = to_map(ratings["userId"].unique())
    item_to_id, id_to_item = to_map(ratings["movieId"].unique())

    rating_matrix = dok_matrix((len(user_to_id), len(item_to_id)))
    for u, i, r in ratings[["userId", "movieId", "rating"]].values:
        rating_matrix[user_to_id[u], item_to_id[i]] = r / 5.0

    return rating_matrix, id_to_user, id_to_item


if __name__ == "__main__":
    pass
