from scipy.sparse import csc_matrix, dok_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sparsesvd import sparsesvd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class SyntheticDataGenerator:
    def __init__(self, ratings):
        self.ratings = ratings

    def build(self):
        raise NotImplementedError()

    def generate(self, n_users):
        raise NotImplementedError()


class MFDataGenerator(SyntheticDataGenerator):
    def __init__(self, ratings, verbose=True):
        super().__init__(ratings)

        self.n_users = self.ratings.shape[0]
        self.n_items = self.ratings.shape[1]
        self.item_ids = list(range(self.n_items))

        self.user_vectors = None
        self.item_vectors = None
        self.gmm = None
        self.item_proba_vectors = None
        self.item_proba_intercepts = None
        self.consider_items = None
        self.verbose = verbose

    def _build_mf(self, components):
        u, s, v = sparsesvd(self.ratings, components)

        self.user_vectors = u.T
        self.item_vectors = np.dot(np.diag(s), v).T

    def _build_gmm(self, gmm_clusters):
        self.gmm = GaussianMixture(gmm_clusters, verbose=2, verbose_interval=1)
        self.gmm.fit(self.user_vectors)

    def _build_implicit_mf(self, consider_items):
        item_proba_vectors = []
        item_proba_intercepts = []

        for i in range(consider_items):
            if self.verbose and (i % 1000 == 0):
                print(i)
            try:
                item_rating_indices = self.ratings.indices[self.ratings.indptr[i]: self.ratings.indptr[i + 1]]
                item_ratings = np.zeros(self.n_users)
                item_ratings[item_rating_indices] = 1.0

                weights = self.n_items * item_ratings / item_ratings.sum() + 1.0

                m = LogisticRegression(max_iter=2000)
                m.fit(self.user_vectors, item_ratings, weights)

                item_proba_vectors.append(m.coef_)
                item_proba_intercepts.append(m.intercept_)
            except Exception as e:
                print(i)
                raise e

        self.item_proba_vectors = np.array(item_proba_vectors)[:, 0, :]
        self.item_proba_intercepts = np.array(item_proba_intercepts)[:, 0]

    def build(self, components=200, gmm_clusters=10, consider_items=3000):
        self.consider_items = consider_items

        self._build_mf(components)
        self._build_implicit_mf(consider_items)
        self._build_gmm(gmm_clusters)

    def _sample_users(self, n_users):
        avg_ratings_per_user = len(self.ratings.indices) / self.n_users

        proper_users = np.where(np.array(self.ratings.sum(axis=1) > 50))

        user_vectors, _ = self.gmm.sample(n_users)
        user_vectors = np.array(np.random.choice(proper_users, size=n_users, replace=True))
        ratings_per_user = np.random.exponential(avg_ratings_per_user, size=n_users)

        return user_vectors, ratings_per_user

    def _sample_user_items(self, user_vector, n_items):
        dotproducts = np.dot(self.item_proba_vectors, user_vector) + self.item_proba_intercepts
        logits = 1.0 / (1e-7 + np.exp(-dotproducts))
        logits = logits / np.sum(logits)

        sampled_items = np.random.choice(self.item_ids, n_items, False, p=logits)

        return sampled_items

    def _get_user_rating(self, user_vector, item):
        sampled_rating = user_vector.dot(self.item_vectors[item])
        return sampled_rating

    def generate(self, n_users, use_actual_user_vectors=False, use_actual_items=False):
        user_vectors, ratings_per_user = self._sample_users(n_users)

        n_items = self.item_vectors.shape[0]
        rating_matrix = dok_matrix((n_users, n_items))

        for u in range(n_users):
            ratings_count = int(ratings_per_user[u])
            ratings_count = min(ratings_count, n_items)

            if use_actual_user_vectors:
                v = self.user_vectors[u, :]
            else:
                v = user_vectors[u, :]

            if use_actual_items:
                sampled_items = self._sample_user_items(v, ratings_count)
            else:
                sampled_items = self._sample_user_items(v, ratings_count)

            for i in sampled_items:
                rating_matrix[u, i] = self._get_user_rating(v, i)
                rating_matrix[u, i] = np.minimum(1.0, np.maximum(rating_matrix[u, i], -1.0))

        return rating_matrix
