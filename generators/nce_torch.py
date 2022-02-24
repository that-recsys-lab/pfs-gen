import math
import torch
from clearml import Task
from torch.nn import BCEWithLogitsLoss

from datasets import Dataset
from generators import TorchDataGenerator
import numpy as np
from torch import Tensor, LongTensor, FloatTensor
from .alias_sampler import AliasMultinomial


class NCESamplerGenerator(TorchDataGenerator):
    def __init__(self, verbose):
        self.noise_logits = None
        self.noise_ratio = None
        self.sampler = None  # type: AliasMultinomial
        self.bce_with_logits = BCEWithLogitsLoss()
        super().__init__(verbose)

    def build(self, task: Task, base_dataset: Dataset, noise_pop_power=2.5, noise_ratio=10, **kwargs):
        noise_logits = np.power((base_dataset.rating_matrix != 0).sum(axis=0), noise_pop_power)
        noise_logits = np.squeeze(np.asarray(noise_logits))

        probas = noise_logits / np.sum(noise_logits)
        probas = probas.clip(min=1e-6)
        probas = probas / np.sum(probas)
        probas = FloatTensor(probas)

        self.sampler = AliasMultinomial(probas)
        self.noise_logits = probas.log()
        self.noise_ratio = noise_ratio
        self._implicit_coef = 1.0

        super().build(task, base_dataset, **kwargs)

    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities
        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution
        Returns:
            - loss: a mis-classification loss for every single case
        """

        # NOTE: prob <= 1 is not guaranteed
        logit_model = torch.cat([logit_target_in_model.unsqueeze(1), logit_noise_in_model], dim=1)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(1), logit_noise_in_noise], dim=1)

        # predicted probability of the word comes from true data distribution
        # The posterior can be computed as following
        # p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
        # For numeric stability we compute the logits of true label and
        # directly use bce_with_logits.
        # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss
        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

        label = torch.zeros_like(logit_model)
        label[:, 0] = 1

        loss = self.bce_with_logits(logit_true, label).mean()
        return loss

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

        rated_prediction, rated_items_logits = self.model(u_tensor, i_tensor)
        rating_loss = ((gt_rating - rated_prediction) * (gt_rating - rated_prediction)).mean()

        negative_samples = self.sampler.draw(len(batch_u), self.noise_ratio)
        negative_samples = LongTensor(negative_samples)
        noise_noise_logits = self.noise_logits[negative_samples]

        noise_model_logits = self.noise_logits[i_tensor]
        _, model_noise_logits = self.model(
            u_tensor,
            negative_samples.transpose(0, 1)
        )
        model_noise_logits = model_noise_logits.transpose(0, 1)

        nce_loss = self.nce_loss(rated_items_logits.flatten(), model_noise_logits, noise_noise_logits, noise_model_logits.flatten())

        loss = rating_loss + nce_loss * logloss_weight

        loss_components = {}
        if return_loss_components:
            loss_components = {
                "nce_loss": nce_loss.item(),
                "rating": rating_loss.item(),
                "total": loss.item(),
            }
        return loss, loss_components

Heatmap for item co-occurence map
Content information

