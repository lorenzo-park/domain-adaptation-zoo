import hydra

import pytorch_lightning as pl

from utils.model import get_model
from utils.dataset import get_datamodule


@hydra.main(config_path=".", config_name="common")
def run(config):
    pl.seed_everything(config.seed)

    if config.logger:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=f"{config.project}",
            name=f"{config.model.backbone}"
        )
    else:
        logger = pl.loggers.TestTubeLogger(
            "output",
            name=f"{config.project}"
        )
        logger.log_hyperparams(config)

    datamodule = get_datamodule(config.dataset)
    model = get_model(config, len(datamodule.CLASSES))
    trainer = pl.Trainer(
        precision=16,
        auto_lr_find=True if config.lr_finder else None,
        deterministic=True,
        check_val_every_n_epoch=1,
        gpus=config.gpus,
        logger=logger,
        max_epochs=config.training.epoch,
        weights_summary="top",
    )

    if config.lr_finder:
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-8, max_lr=1e-1, num_training=100)
        model.hparams.lr = lr_finder.suggestion()
        print(model.hparams.lr)
    else:
        trainer.fit(model, datamodule=datamodule)
        trainer.test()


if __name__ == '__main__':
  run()
