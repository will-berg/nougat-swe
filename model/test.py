from lightning_module import NougatModelPLModule
from transformers import NougatProcessor, VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dataset import NougatDataset
import config as cfg


def test():
    config = VisionEncoderDecoderConfig.from_pretrained(cfg.repo)
    processor = NougatProcessor.from_pretrained(cfg.repo)
    tokenizer = processor.tokenizer
    model = VisionEncoderDecoderModel.from_pretrained(cfg.repo, config=config)

    test_dataset = NougatDataset(dataset_path=cfg.test_path, max_length=cfg.max_length, split="test")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model_module = NougatModelPLModule(config=config, processor=processor, model=model)
    wandb_logger = WandbLogger(project="nougat-swe", name="")

    trainer = pl.Trainer(logger=wandb_logger)
    trainer = pl.Trainer()
    trainer.test(model=model_module, dataloaders=test_dataloader)


if __name__ == "__main__":
    test()
