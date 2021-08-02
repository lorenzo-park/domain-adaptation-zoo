from dataset.pl_officehome import OfficeHomeDataModule


def get_datamodule(config_dataset):
    if config_dataset.name == "officehome":
        dm = OfficeHomeDataModule(
            root=config_dataset.root,
            batch_size=config_dataset.batch_size,
            num_workers=config_dataset.num_workers,
            src_task=config_dataset.src_task,
            tgt_task=config_dataset.tgt_task,
            download=True,
        )
        dm.prepare_data()
        dm.setup()
        return dm