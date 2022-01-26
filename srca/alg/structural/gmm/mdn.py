"""
This file is edited from `oracle/learn/mdn.py` in
    [the open-source code](https://github.com/perfd/perfd).

Reference:
[1] Silvery Fu, Saurabh Gupta, Radhika Mittal, and Sylvia Ratnasamy. On the Use of
    ML for Blackbox System Performance Prediction. NSDI 2021
"""
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch import distributions as D
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .base import GMM
from .base import GMMPredictor


class _MDN(nn.Module):
    def __init__(self, num_input, neurons: int = 50, components: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._base = nn.Sequential(
            nn.Linear(num_input, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
        )
        self._alphas = nn.Linear(neurons, components)
        self._mus = nn.Linear(neurons, components)
        self._sigmas = nn.Linear(neurons, components)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate model output
        """
        latent = self._base(data)
        alphas = F.softmax(self._alphas(latent), dim=-1)
        mus = self._mus(latent)
        sigmas = F.elu(self._sigmas(latent)) + 1 + 1e-6
        return alphas, mus, sigmas


class MDN(pl.LightningModule):
    # pylint: disable=abstract-method,arguments-differ,too-many-ancestors
    """
    Mixture Density Network
    """

    def __init__(self, learning_rate: float = 1e-3, **kwargs):
        super().__init__()
        self._model = _MDN(**kwargs)
        self._learning_rate = learning_rate

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._model(data)

    def loss(self, data: torch.Tensor, target: torch.Tensor):
        """
        Calculate loss for MDN
        """
        alphas, mus, sigmas = self.forward(data)
        comp = D.Normal(loc=mus, scale=sigmas)
        mix = D.Categorical(alphas)
        gmm = D.MixtureSameFamily(mix, comp)
        # ATTENTION: For multi-dimension output, we need transpose target
        likelihood = gmm.log_prob(target)
        return -likelihood.mean()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        data, target = batch
        loss = self.loss(data=data, target=target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        data, target = batch
        loss = self.loss(data=data, target=target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)


class TorchGMM(GMM):
    """
    GMM with torch
    """

    def __init__(
        self,
        alphas: torch.Tensor,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        comp = D.Normal(loc=mus, scale=sigmas)
        mix = D.Categorical(alphas)
        gmm = D.MixtureSameFamily(mix, comp)
        self._model = gmm

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return self._model.sample((n_samples,)).numpy()


class MDNPredictor(GMMPredictor):
    """
    Regress with a Mixture Density Network
    """

    DEFAULT_TRAIN_PARAMS = dict(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=1000,
    )

    def __init__(
        self,
        batch_size: int = 128,
        train_ratio: float = 0.8,
        train_params: dict = None,
        model_params: dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._train_ratio = train_ratio
        if train_params is None:
            train_params = {}
        self._train_params = {**self.DEFAULT_TRAIN_PARAMS, **train_params}
        self._model_params = model_params if model_params else {}
        self._model: MDN = None

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Train the model
        """
        num_samples, num_input = train_x.shape
        num_train = int(num_samples * self._train_ratio)
        data = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
        data_train, data_val = random_split(
            data,
            lengths=[num_train, num_samples - num_train],
            generator=torch.Generator().manual_seed(self._seed) if self._seed else None,
        )

        self._model = MDN(num_input=num_input, **self._model_params)
        trainer = pl.Trainer(
            callbacks=[
                EarlyStopping(monitor="val_loss", min_delta=0, patience=5, mode="min")
            ],
            **self._train_params,
        )
        trainer.fit(
            self._model,
            train_dataloaders=DataLoader(data_train),
            val_dataloaders=DataLoader(data_val),
        )

    def predict(self, test_x: np.ndarray) -> List[GMM]:
        """
        Generate GMM
        """
        data_test = DataLoader(
            TensorDataset(torch.FloatTensor(test_x)), batch_size=self._batch_size
        )
        gmms: List[GMM] = []
        for (data,) in data_test:
            output = self._model.forward(data)
            for alphas, mus, sigmas in zip(*[item.cpu() for item in output]):
                gmms.append(TorchGMM(alphas=alphas, mus=mus, sigmas=sigmas))
        return gmms
