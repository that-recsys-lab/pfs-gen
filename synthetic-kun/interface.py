from scipy.sparse import csc_matrix, dok_matrix
from sklearn.linear_model import LogisticRegression
from sparsesvd import sparsesvd
import numpy as np
from sklearn.mixture import GaussianMixture


class SyntheticDataGenerator:
    def __init__(self, ratings):
        self.ratings = ratings

    def build(self):
        raise NotImplementedError()

    def generate(self, n_users):
        raise NotImplementedError()


class MFDataGenerator(SyntheticDataGenerator):
    def __init__(self, ratings):
        super().__init__(ratings)
        self.ratings = csc_matrix(self.ratings)
        self.n_users = self.ratings.shape[0]
        self.n_items = self.ratings.shape[1]
        self.user_vectors = None
        self.item_vectors = None
        self.gmm = None
        self.item_proba_vectors = None
        self.item_proba_intercepts = None
        self.consider_items = None

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
            item_ratings = (self.ratings[:, i] > 0).todense() * 1.0
            item_ratings = np.squeeze(item_ratings.__array__())
            weights = self.n_items * item_ratings / item_ratings.sum() + 1.0

            m = LogisticRegression(max_iter=2000)
            m.fit(self.user_vectors, item_ratings, weights)

            item_proba_vectors.append(m.coef_)
            item_proba_intercepts.append(m.intercept_)

        self.item_proba_vectors = np.array(item_proba_vectors)[:, 0, :]
        self.item_proba_intercepts = np.array(item_proba_intercepts)[:, 0]

    def _sample_users(self, n_users):
        avg_ratings_per_user = len(self.ratings.nonzero()[0]) / self.n_users

        user_vectors, _ = self.gmm.sample(n_users)
        ratings_per_user = np.random.exponential(avg_ratings_per_user, size=n_users)

        return user_vectors, ratings_per_user

    def build(self, components=200, gmm_clusters=10, consider_items=3000):
        self.consider_items = consider_items

        self._build_mf(components)
        self._build_gmm(gmm_clusters)
        self._build_implicit_mf(consider_items)

    def generate(self, n_users):
        user_vectors, ratings_per_user = self._sample_users(n_users)

        rating_matrix = dok_matrix((n_users, self.consider_items))

        ids = list(range(self.consider_items))

        for u in range(n_users):
            ratings_count = int(np.random.exponential(ratings_per_user[u]))
            ratings_count = min(ratings_count, self.consider_items)

            v = user_vectors[u, :]

            logits = np.exp(np.dot(self.item_proba_vectors, v) + self.item_proba_intercepts)
            logits = logits / np.sum(logits)

            sampled_items = np.random.choice(ids, ratings_count, False, p=logits)
            sampled_rating = self.item_vectors[sampled_items].dot(v)
            sampled_rating = np.minimum(1.0, np.maximum(sampled_rating, 0.0))

            for i, r in zip(sampled_items, sampled_rating):
                rating_matrix[u, i] = r

        return rating_matrix
