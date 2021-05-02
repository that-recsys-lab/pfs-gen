import pandas as pd
from scipy.sparse import dok_matrix, csc_matrix


def to_map(xs):
    map_to = dict(enumerate(xs))
    map_from = {v: k for k, v in map_to.items()}
    return map_from, map_to


class Movielens1MDataset:
    def __init__(self):
        ratings = pd.read_csv("datasets/ml-1m-ratings.dat", sep="::")
        ratings.columns = ["userId", "movieId", "rating", "timestamp"]

        # Map users/items into a more dense structure, as
        # not all movies have ratings
        user_to_id, id_to_user = to_map(ratings["userId"].unique())
        item_to_id, id_to_item = to_map(ratings["movieId"].unique())

        rating_matrix = dok_matrix((len(user_to_id), len(item_to_id)))
        for u, i, r in ratings[["userId", "movieId", "rating"]].values:
            rating_matrix[user_to_id[u], item_to_id[i]] = r

        # That's the only sane way to preserve explicit zeroes
        rating_matrix = csc_matrix(rating_matrix)
        rating_matrix.data = rating_matrix.data / 2.5 - 1.0

        self.rating_matrix = rating_matrix
        self.id_to_user = id_to_user
        self.id_to_item = id_to_item


class AmazonDataset:
    def __init__(self):
        ratings = pd.read_csv("datasets/ratings_Pet_Supplies.csv", header=None)
        ratings.columns = ["userId", "movieId", "rating", "timestamp"]

        # Map users/items into a more dense structure, as
        # not all movies have ratings
        user_to_id, id_to_user = to_map(ratings["userId"].unique())
        item_to_id, id_to_item = to_map(ratings["movieId"].unique())

        rating_matrix = dok_matrix((len(user_to_id), len(item_to_id)))
        for u, i, r in ratings[["userId", "movieId", "rating"]].values:
            rating_matrix[user_to_id[u], item_to_id[i]] = r

        # That's the only sane way to preserve explicit zeroes
        rating_matrix = csc_matrix(rating_matrix)
        rating_matrix.data = rating_matrix.data / 2.5 - 1.0

        self.rating_matrix = rating_matrix
        self.id_to_user = id_to_user
        self.id_to_item = id_to_item


name_to_class = {"ml1m": Movielens1MDataset, "amazon": AmazonDataset}

