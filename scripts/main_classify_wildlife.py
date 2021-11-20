"""
Main script to train and classify the clothes from FashionMNIST
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.wildlife import WildlifeDataModule
from zero_to_hero.models.wildlife_classifier import WildlifeClassifier
from zero_to_hero.trainer.wildlife_with_callbacks import (
    get_wildlife_trainer_with_callbacks,
)

warnings.filterwarnings("ignore")


def main() -> None:
    """
    Implementation of the main function that trains and test a classifier on the FashionMNIST data
    :return:
    """
    pl.seed_everything(
        seed=1,
        workers=True,
    )

    config = read_config(path=Path("../configs/wildlife.yml"))

    datamodule = WildlifeDataModule(config=config)
    datamodule.setup()

    model = WildlifeClassifier(configs=config)

    trainer = get_wildlife_trainer_with_callbacks(config=config)

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    print("Best checkpoint path:", trainer.checkpoint_callback.best_model_path)

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        verbose=True,
    )


if __name__ == "__main__":
    main()
