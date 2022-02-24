import inspect
import typing
from typing import Type, Iterable, TypeVar, Generic, Callable, Iterator, Dict, List
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
    def __init__(self, rating_df=None, rating_matrix=None, rescale=True):
        self.rating_matrix = None  # type:csc_matrix
        self.id_to_user = None
        self.id_to_item = None
        self.n_ratings = 0
        if rating_matrix is not None:
            self.rating_matrix = rating_matrix
            self.id_to_user = {i: str(i) for i in range(self.rating_matrix.shape[0])}
            self.id_to_item = {i: str(i) for i in range(self.rating_matrix.shape[1])}
        elif rating_df is not None:
            # Map users/items into a more dense structure, as
            # not all movies have ratings
            user_to_id, id_to_user = to_map(rating_df["userId"].unique())
            item_to_id, id_to_item = to_map(rating_df["movieId"].unique())

            rating_matrix = dok_matrix((len(user_to_id), len(item_to_id)))
            for u, i, r in rating_df[["userId", "movieId", "rating"]].values:
                rating_matrix[user_to_id[u], item_to_id[i]] = r

            # That's the only sane way to preserve explicit zeroes
            rating_matrix = csc_matrix(rating_matrix)
            self.rescale = rescale
            if rescale:
                rating_matrix.data = rating_matrix.data / 2.5 - 1.0

            self.rating_matrix = rating_matrix
            self.id_to_user = id_to_user
            self.id_to_item = id_to_item
        else:
            raise Exception("Data, please")
        self.n_users = len(self.id_to_user)
        self.n_items = len(self.id_to_item)
        self.n_ratings = len(rating_matrix.data)

    def write(self, filename):
        with open(filename, "w") as f:
            for k, v in self.rating_matrix.items():
                rating = int(v * 2.5 + 2.5) if self.rescale else v
                f.write("%d,%d,%d\n" % (k[0], k[1], rating))

    def as_iterator(self):
        rows, cols = self.rating_matrix.nonzero()
        for row, col in zip(rows, cols):
            yield row, col, self.rating_matrix[row, col]

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
    def read(filename, sep=None):
        with open(filename, "r") as f:
            df = []
            for line in f:
                args = line.split(sep)
                df.append((args[0], args[1], args[2]))
        df = pd.DataFrame(df, columns=["userId", "movieId", "rating"])
        return Dataset(df)


class Movielens1MDataset(Dataset):
    def __init__(self, path="datasets/ml-1m-ratings.dat", sep="::", rescale=True):
        ratings = pd.read_csv(path, sep=sep)
        ratings.columns = ["userId", "movieId", "rating", "timestamp"]
        super().__init__(ratings, rescale=rescale)


class AmazonDataset(Dataset):
    def __init__(self, path="datasets/ratings_Pet_Supplies.csv", rescale=True):
        ratings = pd.read_csv(path, header=None)
        ratings.columns = ["userId", "movieId", "rating", "timestamp"]
        super().__init__(ratings, rescale=rescale)


__name_to_class = {"ml1m": Movielens1MDataset, "amazon": AmazonDataset}


def get_dataset(name_or_task_id) -> Dataset:
    if name_or_task_id is None:
        raise Exception()
    try:
        gen_task = Task.get_task(task_id=name_or_task_id)
    except ValueError:
        gen_task = None

    if name_or_task_id in __name_to_class:
        return __name_to_class[name_or_task_id]()
    elif gen_task is not None:
        dataset_path = gen_task.artifacts["dataset"].get_local_copy()
        return Dataset.read(dataset_path, sep=",")
    raise Exception("Unknown dataset " + str(name_or_task_id))


T = TypeVar("T")


class CursorHandle(Generic[T]):
    def _generate_getter(self, field):
        def f():
            v = self._prop_fields.get(field)
            if v is None:
                return self._current.get(v)
            elif callable(v):
                return v(self._current)
            else:
                return self._current.get(v)
        return f

    def _generate_props(self, props):
        for prop, getter in props:
            self._prop_fields[prop] = getter
            prop_getter = self._generate_getter(prop)
            prop_getter.__doc__ = f"Property {prop}"
            setattr(self, f"get_{prop}", prop_getter)

    def __init__(self, over_what: Iterator[T], length: int,
                 props_list: List[str] = None,
                 props_dict: Dict[str, Callable or str or int] = None):
        if props_list is None and props_dict is None:
            assert "supply props_list or props_dict!"
        self.idx = -1
        self._collection = over_what
        self._length = length
        self._current = None
        self._prop_fields = {}
        self._props = props_dict.items() if props_dict else zip(props_list, [None] * len(props_list))
        self._generate_props(self._props)

    def __repr__(self):
        flds = [f"{k}:{repr(getattr(self, 'get_' + k)())}" for k in self._prop_fields.keys()]
        return (f"Cursor {hex(id(self))} current entry #{self.idx}: <" + ">, <".join([
            x[:50]+"..." if len(x) > 50 else x
            for x in flds]) +
                ">")

    def reset(self):
        self.idx = -1
        return self

    def __iter__(self):
        return self.reset()

    def __next__(self):
        self.idx += 1
        if self.idx > self._length:
            raise StopIteration()
        self._current = next(self._collection)
        return self

    def __len__(self):
        return self._length

    @property
    def is_valid(self) -> bool:
        return (0 <= self.idx) and ((self._length is None) or (self.idx < self._length))

    @property
    def is_first(self) -> bool:
        return self.idx == 0

    @property
    def is_last(self) -> bool:
        return (self._length is not None) and (self.idx == self._length - 1)


def ratings_batches(ds: Dataset, batch_size: int = 256):
    valid_indices = np.array(ds.rating_matrix.nonzero()).transpose()
    np.random.shuffle(valid_indices)
    batches = np.array_split(ds.n_ratings // batch_size)
    for batch in batches:
        yield ds.rating_matrix[batch[0], batch[1]]
