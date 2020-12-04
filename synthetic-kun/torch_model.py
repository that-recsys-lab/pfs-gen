import torch
from scipy.sparse import csc_matrix, dok_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sparsesvd import sparsesvd
import numpy as np
from sklearn.mixture import GaussianMixture
from interface import MFDataGenerator
import torch.nn


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=200):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.item_implicit_factors = torch.nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        predicted_ratings = (self.user_factors(user) * self.item_factors(item)).sum(2)
        predicted_probas = (self.user_factors(user) * self.item_implicit_factors(item)).sum(2)
        return predicted_ratings, predicted_probas


class TorchDataGenerator(MFDataGenerator):
    def __init__(self, ratings, epochs, verbose=True):
        super().__init__(ratings, verbose)
        self.epochs = epochs
        self.model = None

    def _build_mf(self, components):
        u, s, v = sparsesvd(self.ratings, components)

        self.user_vectors = u.T
        self.item_vectors = np.dot(np.diag(s), v).T

    def _build_torch_model(self, components):
        model = MatrixFactorization(self.n_users, self.n_items, components)
        model.user_factors = torch.nn.Embedding(
            self.n_users, 200, _weight=torch.FloatTensor(self.user_vectors)
        )
        model.item_factors = torch.nn.Embedding(
            self.n_items, 200, _weight=torch.FloatTensor(self.item_vectors)
        )
        self.model = model

    def _train_torch_model(self, logloss_weight=2.0, lr=5e-3, implicit_logloss=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        m2 = dok_matrix(self.ratings)

        lsum, lsum2, lsum3 = [], [], []
        batch_u, batch_i, batch_r = [], [], []
        crossentropy = torch.nn.BCELoss()

        for e in range(self.epochs):
            for j, ((u, i), r) in enumerate(m2.items()):
                if (j > 0) and ((j % 256 == 0) or (j == len(m2.items()))):
                    optimizer.zero_grad()

                    rating = torch.FloatTensor([batch_r])
                    u_tensor = torch.LongTensor([batch_u])
                    i_tensor = torch.LongTensor([batch_i])
                    ones_tensor = torch.FloatTensor([1.0] * len(batch_i))

                    batch_unrated_items = [np.random.randint(0, self.n_items) for _ in range(len(batch_u))]
                    batch_unrated_items_tensor = torch.LongTensor(batch_unrated_items)
                    zeroes_tensor = torch.FloatTensor([0.0] * len(batch_i))

                    predicted_ratings, rated_items_proba = self.model(u_tensor, i_tensor)
                    _, unrated_items_proba = self.model(u_tensor, batch_unrated_items_tensor)

                    if implicit_logloss:
                        rated_items_proba = torch.sigmoid(rated_items_proba)
                        unrated_items_proba = torch.sigmoid(unrated_items_proba)

                        proba_loss = (
                                crossentropy(rated_items_proba, ones_tensor) / 2 +
                                crossentropy(unrated_items_proba, zeroes_tensor) / 2
                        )
                    else:
                        proba_loss = (
                                ((rated_items_proba - ones_tensor) ** 2).mean() / 2 +
                                ((unrated_items_proba - zeroes_tensor) ** 2).mean() / 2
                        )
                    rating_loss = ((rating - predicted_ratings) * (rating - predicted_ratings)).mean()

                    loss = rating_loss + proba_loss * logloss_weight

                    lsum.append(proba_loss.item())
                    lsum2.append(rating_loss.item())
                    lsum3.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    batch_u = []
                    batch_i = []
                    batch_r = []

                batch_u.append(u)
                batch_r.append(r)
                batch_i.append(i)

            if self.verbose:
                print("Epoch ", e, "RMSE", np.mean(lsum2), "Proba loss", np.mean(lsum), "Total loss", np.mean(lsum3))
                lsum, lsum2, lsum3 = [], [], []

    def _build_gmm(self, gmm_clusters):
        self.gmm = GaussianMixture(gmm_clusters, verbose=2, verbose_interval=1)
        self.gmm.fit(self.user_vectors)

    def _build_implicit_mf(self, consider_items):
        pass

    def build(self, components=200, gmm_clusters=10, consider_items=3000):
        self.consider_items = consider_items

        if self.user_vectors is None:
            self._build_mf(components)
        self._build_torch_model(components)
        self._build_gmm(gmm_clusters)

    def _sample_users(self, n_users):
        avg_ratings_per_user = len(self.ratings.indices) / self.n_users

        user_vectors, _ = self.gmm.sample(n_users)
        ratings_per_user = np.random.exponential(avg_ratings_per_user, size=n_users)

        return user_vectors, ratings_per_user

    def _sample_user_items(self, user_vector, n_items):
        # MAGIC HERE
        # We transform arbitrary scores trained with RMSE loss
        # into probabilities
        # we can do this in a bunch of ways
        # hence, sigmoid and * 10 here
        dotproducts = np.dot(self.item_proba_vectors, user_vector) * 10.0
        logits = 1.0 / (1e-7 + np.exp(-dotproducts))
        logits = logits / np.sum(logits)

        sampled_items = np.random.choice(self.item_ids, n_items, False, p=logits)

        return sampled_items

    def _get_user_rating(self, user_vector, item):
        sampled_rating = user_vector.dot(self.item_vectors[item])
        return sampled_rating
