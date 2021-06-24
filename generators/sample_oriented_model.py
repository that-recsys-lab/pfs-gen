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


class Sampletext(SyntheticDataGenerator):
    def __init__(self, verbose=True):
        self._implicit_coef = None
        self.model = None
        self.verbose = verbose

        self.items_view = None
        self.avg_ratings_per_user = None
        self.crossentropy = torch.nn.BCELoss()

        self.item_ids = None
        self.user_ids = None
        self.user_map = None
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

    def _train_torch_model(self, ds: Dataset, task: Task, epochs, logloss_weight, lr,
                           batch_user_size=200,
                           samples_per_user=400,
                           logloss_only_lr=1e-1,
                           epochs_logloss_only=30):
        task.logger.report_text("Training torch model")

        self.model.user_factors.requires_grad_(False)
        self.model.item_factors.requires_grad_(False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=logloss_only_lr)

        loss_components_list = []

        for e in range(epochs):
            if e == epochs_logloss_only:
                self.model.user_factors.requires_grad_(True)
                self.model.item_factors.requires_grad_(True)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            shuffle(self.user_ids)
            batches = max(1, int(len(self.user_ids) / batch_user_size))
            hits = 0

            for batch_user_ids in tqdm(np.array_split(self.user_ids, batches)):
                batch_user_ids_tensor = LongTensor(batch_user_ids)
                user_vectors_batch = self.model.user_factors(batch_user_ids_tensor)
                probas = user_vectors_batch.matmul(self.model.item_implicit_factors.weight.transpose(0, 1))
                probas_raw = probas.detach().numpy()
                probas = torch.sigmoid(probas)

                sampled_items_labels = [0] * len(batch_user_ids) * samples_per_user
                sampled_items_probas = FloatTensor([0] * len(batch_user_ids) * samples_per_user)

                rated_item_userid = []
                rated_item_itemid = []
                rated_item_rating = []

                for user_index_offset in range(len(batch_user_ids)):
                    u = user_index_offset * samples_per_user
                    sampled_items = self._sample_user_items(probas_raw[user_index_offset, :], samples_per_user)

                    sampled_items_probas[u: u + samples_per_user] = probas[user_index_offset, sampled_items]
                    for idx, i in enumerate(sampled_items):
                        if i in self.user_map[batch_user_ids[user_index_offset]]:
                            sampled_items_labels[u + idx] = 1
                            rated_item_userid.append(batch_user_ids[user_index_offset])
                            rated_item_itemid.append(i)
                            rated_item_rating.append(self.user_map[batch_user_ids[user_index_offset]][i])
                            hits += 1
                optimizer.zero_grad()

                sampled_items_labels = FloatTensor(sampled_items_labels)
                loss_crossentropy = self.crossentropy(sampled_items_probas, sampled_items_labels)

                loss_components = {
                    "hits": hits,
                    "Logloss": loss_crossentropy.item(),
                }

                if hits > 0:
                    rated_item_userid = LongTensor(rated_item_userid)
                    rated_item_itemid = LongTensor(rated_item_itemid)
                    rated_item_rating = FloatTensor(rated_item_rating)
                    predicted_ratings = (self.model.user_factors(rated_item_userid) * (self.model.item_factors(rated_item_itemid))).sum(1)
                    mse_loss_part = (predicted_ratings - rated_item_rating) * (predicted_ratings - rated_item_rating)
                    mse_loss_part = mse_loss_part.mean()
                    loss = loss_crossentropy + mse_loss_part / logloss_weight
                    loss_components["mse"] = mse_loss_part.item()
                else:
                    loss = loss_crossentropy
                loss_components["total"] = loss.item()

                loss_components_list.append(loss_components)

                loss.backward()
                optimizer.step()

            if loss_components_list:
                average_metrics = {
                    component: float(np.mean([x.get(component, 0.0) for x in loss_components_list]))
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
              logloss_weight=2.0, lr=5e-3, epochs_logloss_only=25):
        task.set_user_properties(**to_clear_ml_params(locals(), ["task", "self"]))
        s_coo = base_dataset.rating_matrix.tocoo()
        self._implicit_coef = 1.0

        self.user_map = {}
        self.user_ids = []
        self.avg_rating = np.mean(s_coo.data)
        for u, i, r in zip(s_coo.row, s_coo.col, s_coo.data):
            self.user_ids.append(u)
            if u not in self.user_map:
                self.user_map[u] = {}
            self.user_map[u][i] = r - self.avg_rating

        self.user_ids = list(set(self.user_ids))
        self.item_ids = list(range(len(base_dataset.id_to_item)))
        self.avg_ratings_per_user = base_dataset.n_ratings / base_dataset.n_users

        if self.user_vectors is None:
            self._build_mf(base_dataset, task, components)

        self._build_torch_model(base_dataset, task, components)

        self._train_torch_model(base_dataset, task, epochs, logloss_weight=logloss_weight, lr=lr, epochs_logloss_only=epochs_logloss_only)

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
                    rating_matrix[u, i] = self._get_user_rating(v, i) + self.avg_rating
                    rating_matrix[u, i] = np.minimum(1.0, np.maximum(rating_matrix[u, i], -1.0))
        rating_matrix = csc_matrix(rating_matrix)
        ds = Dataset(rating_matrix=rating_matrix)
        return ds
