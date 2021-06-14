import pandas as pd
from clearml import Task
from scipy.sparse import dok_matrix, csc_matrix
import numpy as np

import matplotlib.pyplot as plt


def get_hist(x, title=None, *args, **kwargs):
    f = plt.figure()
    plt.hist(x, *args, **kwargs, figure=f)
    if title is not None:
        plt.title(title)
    return f


def to_map(xs):
    map_to = dict(enumerate(xs))
    map_from = {v: k for k, v in map_to.items()}
    return map_from, map_to


class Dataset:
    def __init__(self, rating_df=None, rating_matrix=None):
        self.rating_matrix = None
        self.id_to_user = None
        self.id_to_item = None
        if rating_matrix is not None:
            self.rating_matrix = rating_matrix
            self.id_to_user = {i: str(i) for i in range(self.rating_matrix.shape[0])}
            self.id_to_item = {i: str(i) for i in range(self.rating_matrix.shape[1])}
            return
        if rating_df is not None:
            # Map users/items into a more dense structure, as
            # not all movies have ratings
            user_to_id, id_to_user = to_map(rating_df["userId"].unique())
            item_to_id, id_to_item = to_map(rating_df["movieId"].unique())

            rating_matrix = dok_matrix((len(user_to_id), len(item_to_id)))
            for u, i, r in rating_df[["userId", "movieId", "rating"]].values:
                rating_matrix[user_to_id[u], item_to_id[i]] = r

            # That's the only sane way to preserve explicit zeroes
            rating_matrix = csc_matrix(rating_matrix)
            rating_matrix.data = rating_matrix.data / 2.5 - 1.0

            self.rating_matrix = rating_matrix
            self.id_to_user = id_to_user
            self.id_to_item = id_to_item

    def write(self, filename):
        with open(filename, "w") as f:
            for k, v in self.rating_matrix.items():
                f.write("%d,%d,%d\n" % (k[0], k[1], int(v * 2.5 + 2.5)))

    def as_iterator(self):
        for k, v in self.rating_matrix.items():
            yield k[0], k[1], v

    def print_stats(self):
        n_users, n_items = len(self.id_to_user), len(self.id_to_item)
        n_ratings = self.rating_matrix.getnnz()

        user_means = np.array(self.rating_matrix.mean(axis=1)).squeeze()
        item_means = np.array(self.rating_matrix.mean(axis=0)).squeeze()

        return [
            ("users", n_users),
            ("items", n_items),
            ("sparsity", 1 - n_ratings / n_users / n_items),
            ("mean rating", np.mean(self.rating_matrix)),

            ("user avg rating count", n_ratings / n_users),
            ("item avg rating count", n_ratings / n_items),

            ("rating", get_hist(self.rating_matrix.data, title="Ratings", bins=10)),
            ("user rating distribution", get_hist(user_means, title="User mean ratings", bins=10)),
            ("item rating distribution", get_hist(item_means, title="Item mean ratings", bins=10)),
        ]

    @staticmethod
    def read(filename):
        with open(filename, "r") as f:
            df = []
            for line in f:
                args = line.split()
                df.append((args[0], args[1], args[2]))
        df = pd.DataFrame(df, columns=["userId", "movieId", "rating"])
        return Dataset(df)

    @staticmethod
    def read_clearml_experiment(task_id, artifact_name="dataset"):
        gen_task = Task.get_task(task_id=task_id)
        dataset_path = gen_task.artifacts[artifact_name].get_local_copy()
        return Dataset.read(dataset_path)


class Movielens1MDataset(Dataset):
    def __init__(self):
        ratings = pd.read_csv("datasets/ml-1m-ratings.dat", sep="::")
        ratings.columns = ["userId", "movieId", "rating", "timestamp"]
        super().__init__(ratings)


class AmazonDataset(Dataset):
    def __init__(self):
        ratings = pd.read_csv("datasets/ratings_Pet_Supplies.csv", header=None)
        ratings.columns = ["userId", "movieId", "rating", "timestamp"]
        super().__init__(ratings)


name_to_class = {"ml1m": Movielens1MDataset, "amazon": AmazonDataset}

