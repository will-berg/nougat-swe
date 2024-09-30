from nltk import edit_distance
import numpy as np
import re
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

import config as cfg
from util import compute_metrics


class NougatModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        # Store the distribution of the metrics in the test steps in a list
        self.test_metrics = []

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        # Forward pass through the model
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)

        sch = self.lr_schedulers()
        # step every N batches
        if (batch_idx + 1) % 15 == 0:
            sch.step()

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, _, target_sequences = batch
        batch_size = pixel_values.shape[0]
        # Feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=cfg.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # Remove first task start token
            predictions.append(seq)

        scores = []
        for pred, seq in zip(predictions, target_sequences):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            seq = seq.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, seq) / max(len(pred), len(seq)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    # Test step for one batch
    def test_step(self, batch, batch_idx):
        pixel_values, _, target_sequences = batch
        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=cfg.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            predictions.append(seq)

        scores = []
        for pred, seq in zip(predictions, target_sequences):
            # Compute the metrics for the current prediction, returns a dict
            metrics = compute_metrics(pred, seq)
            scores.append([metrics["edit_dist"], metrics["bleu"], metrics["meteor"], metrics["precision"], metrics["recall"], metrics["f_measure"]]) # , metrics["wer"], metrics["cer"]])

        # Append the scores to the test_metrics list
        self.test_metrics.append(scores)
        # Save the test_metrics list to a numpy file, overwriting the previous file until the final batch
        np.save("test_metrics.npy", np.array(self.test_metrics))
        scores = np.array(scores)
        scores_dict = {
            "edit_distance": np.nanmean(scores[:, 0]),
            "bleu": np.nanmean(scores[:, 1]),
            "meteor": np.nanmean(scores[:, 2]),
            "precision": np.nanmean(scores[:, 3]),
            "recall": np.nanmean(scores[:, 4]),
            "f_measure": np.nanmean(scores[:, 5]),
            # "wer": np.mean(scores[:, 6]),
            # "cer": np.mean(scores[:, 7]),
        }
        # Print the scores dict, each key-value pair on a new line
        for key, value in scores_dict.items():
            print(f"{key}: {value}")

        self.log_dict(scores_dict)

        return scores

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values, _, target_sequences = batch
        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=cfg.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            predictions.append(seq)
        return predictions

    def configure_optimizers(self):
        lr_end = 7.5e-6
        lr_init = self.config.get("lr", 5e-5)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_init)

        # Define the lambda function for the scheduler
        def lr_lambda(current_step):
            factor = 0.9996 ** (current_step // 15)
            lr = max(lr_end, lr_init * factor)
            return lr / lr_init  # Scale by initial lr

        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [scheduler]

    """
    def configure_optimizers(self):
        # Could also add a learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

        return optimizer
    """