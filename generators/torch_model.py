from random import shuffle
from typing import Type

from clearml import Task
from scipy.sparse import dok_matrix, csc_matrix
from sparsesvd import sparsesvd
import numpy as np
from sklearn.mixture import GaussianMixture
from torch import Tensor, FloatTensor, LongTensor
from tqdm import tqdm

from datasets import Dataset
from tools import to_clear_ml_params
from .interface import SyntheticDataGenerator
import torch.nn


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=200):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.item_implicit_factors = torch.nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        predicted_ratings = (self.user_factors(user) * self.item_factors(item)).sum(2)
        predicted_logits = (self.user_factors(user) * self.item_implicit_factors(item)).sum(2)
        return predicted_ratings, predicted_logits

    def user_choice_probas(self, v):
        formula = Tensor(v).matmul(self.item_implicit_factors.weight.transpose(0, 1))

        return formula.detach().numpy()


class TorchDataGenerator(SyntheticDataGenerator):
    def __init__(self, verbose=True):
        self._implicit_coef = None
        self.model = None
        self.verbose = verbose

        self.items_view = None
        self.avg_ratings_per_user = None

        self.item_ids = None
        self.user_vectors = None
        self.item_vectors = None
        self.gmm = None

    def _build_mf(self, ds: Dataset, task: Task, components):
        task.logger.report_text("Training MF")
        u, s, v = sparsesvd(ds.rating_matrix, components)

        self.user_vectors = u.T
        self.item_vectors = np.dot(np.diag(s), v).T

    def _build_torch_model(self, ds: Dataset, task: Task, components):
        task.logger.report_text("Building torch model")

        model = MatrixFactorization(ds.n_users, ds.n_items, components)
        model.user_factors = torch.nn.Embedding(
            ds.n_users, components, _weight=FloatTensor(self.user_vectors)
        )
        model.item_factors = torch.nn.Embedding(
            ds.n_items, components, _weight=FloatTensor(self.item_vectors)
        )
        self.model = model

    def _sample_unrated_items(self, ds: Dataset, batch_u, batch_i, batch_r):
        return [np.random.randint(0, ds.n_items) for _ in range(len(batch_u))]

    def _construct_loss(
            self,
            ds: Dataset,
            batch_r, batch_u, batch_i,
            logloss_weight,
            return_loss_components
    ):
        gt_rating = FloatTensor([batch_r])
        u_tensor = LongTensor([batch_u])
        i_tensor = LongTensor([batch_i])

        batch_unrated_items = self._sample_unrated_items(ds, batch_u, batch_i, batch_r)
        batch_unrated_items_tensor = LongTensor(batch_unrated_items)

        rated_prediction, rated_items_logits = self.model(u_tensor, i_tensor)
        unrated_prediction, unrated_items_logits = self.model(u_tensor, batch_unrated_items_tensor)

        ones_tensor = torch.ones_like(rated_items_logits)
        zeroes_tensor = torch.zeros_like(unrated_items_logits)

        proba_loss = (
                ((rated_items_logits - ones_tensor) ** 2).mean() / 2 +
                ((unrated_items_logits - zeroes_tensor) ** 2).mean() / 2
        )
        rating_loss = ((gt_rating - rated_prediction) * (gt_rating - rated_prediction)).mean()
        loss = rating_loss + proba_loss * logloss_weight

        loss_components = {}
        if return_loss_components:
            loss_components = {
                "proba": proba_loss.item(),
                "rating": rating_loss.item(),
                "total": loss.item(),
            }
        return loss, loss_components

    def _train_torch_model(self, ds: Dataset, task: Task, epochs, logloss_weight, lr, batch_size):
        task.logger.report_text("Training torch model")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_components_list = []
        batch_u, batch_i, batch_r = [], [], []

        if self.items_view is None:
            m2 = dok_matrix(ds.rating_matrix)
            self.items_view = list(m2.items())

        for e in range(epochs):
            shuffle(self.items_view)
            for j, ((u, i), r) in enumerate(tqdm(self.items_view)):
                if (len(batch_u) > 0) and (j % batch_size == 0):  # or j == len(m2.items())):
                    optimizer.zero_grad()

                    loss, loss_components = self._construct_loss(
                        ds, batch_r, batch_u, batch_i,
                        logloss_weight,
                        self.verbose
                    )

                    loss_components_list.append(loss_components)

                    loss.backward()
                    optimizer.step()
                    batch_u = []
                    batch_i = []
                    batch_r = []

                batch_u.append(u)
                batch_r.append(r)
                batch_i.append(i)

            if loss_components_list:
                average_metrics = {
                    component: float(np.mean([x[component] for x in loss_components_list]))
                    for component in loss_components_list[0]
                }

                if self.verbose:
                    print("Epoch ", e, "\t".join("%s: %f" % (k, v) for k, v in average_metrics.items()))

                for component in average_metrics:
                    task.logger.report_scalar(
                        "Loss", component, average_metrics[component], e
                    )

    def _build_gmm(self, task: Task, gmm_clusters):
        task.logger.report_text("Building GMM")
        self.gmm = GaussianMixture(gmm_clusters, verbose=2, verbose_interval=1)
        self.gmm.fit(self.user_vectors)

    def build(self, task: Task, base_dataset: Dataset, epochs=50, components=200, gmm_clusters=10,
              logloss_weight=2.0, lr=5e-3, batch_size=5000):
        task.set_user_properties(**to_clear_ml_params(locals(), ["task", "self"]))
        self.item_ids = list(range(len(base_dataset.id_to_item)))
        self.avg_ratings_per_user = base_dataset.n_ratings / base_dataset.n_users

        if self.user_vectors is None:
            self._build_mf(base_dataset, task, components)

        self._build_torch_model(base_dataset, task, components)

        self._train_torch_model(base_dataset, task, epochs, logloss_weight=logloss_weight, lr=lr, batch_size=batch_size)

        self._build_gmm(task, gmm_clusters)

    def _sample_users(self, n_users):
        user_vectors, _ = self.gmm.sample(n_users)
        ratings_per_user = np.random.exponential(self.avg_ratings_per_user, size=n_users)

        return user_vectors, ratings_per_user

    def _sample_user_items(self, dotproducts, n_items):
        # MAGIC HERE
        # We transform arbitrary scores trained with RMSE loss
        # into probabilities
        # we can do this in a bunch of ways
        # hence, sigmoid and * 10 here
        dotproducts = dotproducts * self._implicit_coef
        logits = 1.0 / (1e-7 + np.exp(-dotproducts))
        logits = logits / np.sum(logits)

        sampled_items = np.random.choice(self.item_ids, n_items, False, p=logits)

        return sampled_items

    def _get_user_rating(self, user_vector, item):
        sampled_rating = user_vector.dot(self.item_vectors[item])
        return sampled_rating

    def generate(self, task, n_users=None, use_actual_user_vectors=False, use_actual_item_choice=False,
                 implicit_coef=15.0, **kwargs):
        task.set_user_properties(**to_clear_ml_params(locals(), ["task", "self"]))
        self._implicit_coef = implicit_coef

        if n_users is None:
            n_users = self.user_vectors.shape[0]

        user_vectors, ratings_per_user = self._sample_users(n_users)

        n_items = self.item_vectors.shape[0]
        rating_matrix = dok_matrix((n_users, n_items))

        batches = max(1, int(n_users / 1000))
        for batch_n, user_vectors_batch in enumerate(tqdm(np.array_split(user_vectors, batches))):
            user_index_base = batch_n * 1000

            # Super-efficient batched matrix multiplication that exploits pytorch (=GPU)
            probas = self.model.user_choice_probas(user_vectors_batch)

            for user_index_offset in range(len(probas)):
                u = user_index_base + user_index_offset

                ratings_count = int(ratings_per_user[u])
                ratings_count = min(ratings_count, n_items)

                v = user_vectors[user_index_offset, :]

                sampled_items = self._sample_user_items(probas[user_index_offset, :], ratings_count)

                for i in sampled_items:
                    rating_matrix[u, i] = self._get_user_rating(v, i)
                    rating_matrix[u, i] = np.minimum(1.0, np.maximum(rating_matrix[u, i], -1.0))
        rating_matrix = csc_matrix(rating_matrix)
        ds = Dataset(rating_matrix=rating_matrix)
        return ds
