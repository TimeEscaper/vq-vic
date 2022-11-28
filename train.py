import os
import fire
import logging
import nip
import pytorch_lightning as pl

from datetime import datetime
from torch.utils.data import DataLoader
from nip.elements import Element
from pathlib import Path
from typing import Dict
from pytorch_lightning.loggers import NeptuneLogger

import lib


_SEED = 42

_KEY_EXPERIMENT_NAME = "experiment_name"


def _train(trainer: pl.Trainer,
           pl_module: pl.LightningModule,
           train_dataloader: DataLoader,
           val_dataloader: DataLoader,
           experiment_params: Dict,
           config: Element,
           **_):
    if type(trainer.logger) == NeptuneLogger:
        datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if _KEY_EXPERIMENT_NAME in experiment_params:
            experiment_name = "_" + experiment_params[_KEY_EXPERIMENT_NAME]
        else:
            experiment_name = ""

        config_name = "config_" + datetime_now + experiment_name + ".yaml"
        config_path = Path(experiment_params['path']) / config_name
        nip.dump(config_path, config)  # mb: dump into version
        trainer.logger.experiment["nip_config"].upload(str(config_path))
    else:
        config_path = Path(experiment_params['path']) / 'config.nip'
        nip.dump(config_path, config)  # mb: dump into version

    trainer.fit(pl_module, train_dataloader, val_dataloader)


def train(config: str):
    """Runs training based on config.

    Parameters
    ----------
    config:
        Path to config file.
    """
    pl.seed_everything(_SEED)
    nip.run(config, _train,
            verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == "__main__":
    fire.Fire(train)
