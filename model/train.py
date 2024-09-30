from transformers import VisionEncoderDecoderConfig
from transformers import NougatProcessor, VisionEncoderDecoderModel

from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    GradientAccumulationScheduler,
    Callback,
	EarlyStopping
)
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import config as cfg
from dataset import NougatDataset
from lightning_module import NougatModelPLModule


class PushToHubCallback(Callback):
    def __init__(self, push_every_x_epochs=5, repo=cfg.repo):
        self.push_every_x_epochs = push_every_x_epochs
        self.repo = repo

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.push_every_x_epochs == 0:
            print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
            pl_module.model.push_to_hub(self.repo,
                                        commit_message=f"Epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(self.repo)
        pl_module.model.push_to_hub(self.repo,
                                    commit_message=f"Training done")

def train():
    config = VisionEncoderDecoderConfig.from_pretrained(cfg.base_model)
    # Option to modify config here by setting specific values
    # config.encoder.image_size = image_size # (height, width)

    processor = NougatProcessor.from_pretrained(cfg.base_model)
    tokenizer = processor.tokenizer
    model = VisionEncoderDecoderModel.from_pretrained(cfg.base_model, config=config)

    train_dataset = NougatDataset(dataset_path=cfg.train_path, max_length=cfg.max_length, split="train")
    val_dataset = NougatDataset(dataset_path=cfg.val_path, max_length=cfg.max_length, split="validation")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    def custom_collate_fn(batch):
        # batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)

    config = {
        "max_epochs":2,
        "max_steps":-1,
        # "val_check_interval":0.2, # How many times we want to validate during an epoch
        "check_val_every_n_epoch":4,
        "gradient_clip_val":0.5,
        "num_training_samples_per_epoch": 552,
        "lr":5e-5,
        "train_batch_sizes": [2],
        "val_batch_sizes": [2],
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 100, # old: 800/8*30/10, 10%
        "result_path": "./result",
        "verbose": True,
    }

    model_module = NougatModelPLModule(config, processor, model)

    # Need to login to wandb and the hub before running this: "wandb login" and "huggingface-cli login"
    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    lr_callback = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(project="nougat-swe", name="eng-swe-run")

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices="auto",
        # strategy="ddp_find_unused_parameters_true",
        accelerator="auto",
        max_epochs=config.get("max_epochs"),
        # max_steps=config.get("max_steps"),
        # val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[
            PushToHubCallback(),
            early_stop_callback,
            lr_callback,
			# GradientAccumulationScheduler({0: 3})
        ]
    )

    trainer.fit(model=model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    train()
