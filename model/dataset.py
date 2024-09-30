from typing import Tuple
import logging
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image, UnidentifiedImageError
import unicodedata
from transformers import NougatProcessor
import torch
import orjson
from torch.utils.data import Dataset

import config as cfg


class SciPDFDataset(Dataset):
    """
    Custom dataset for scientific PDF data.

    This dataset loads data from JSONL files and provides access to images, ground truth,
    and metadata.

    Args:
        path_to_index (str): Path to the index file.
        split (str, optional): Split of the dataset (e.g., "train", "test"). Default is "train".
        root_name (str, optional): Root directory name. Default is an empty string.
        template (str, optional): Template for split naming. Default is "%s".

    Attributes:
        empty_sample: Placeholder for empty samples.
    """

    empty_sample = None

    def __init__(
        self,
        path_to_index: str,
        split: str = "train",
        root_name="",
        template="%s",
    ) -> None:
        super().__init__()
        self.path_to_index = Path(path_to_index)
        self.root_name = root_name
        self.path_to_root = self.path_to_index.parent
        if not split in self.path_to_index.stem:
            pti = self.path_to_root / (template % split + ".jsonl")
            if pti.exists():
                self.path_to_index = pti
            else:
                raise ValueError(f'Dataset file for split "{split}" not found: {pti}')
        self.dataset_file = None  # mulitprocessing
        # load seek map
        seek_path = self.path_to_root / (self.path_to_index.stem + ".seek.map")
        if seek_path.exists():
            self.seek_map = orjson.loads(seek_path.open().read())
        else:
            raise ValueError(
                'No "%s" found in %s' % (seek_path.name, str(self.path_to_root))
            )
        self.dataset_length = len(self.seek_map)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict:
        position = self.seek_map[index]
        if self.dataset_file is None:
            self.dataset_file = self.path_to_index.open()
        self.dataset_file.seek(position)
        line = self.dataset_file.readline()
        try:
            data: Dict = orjson.loads(line)
        except Exception as e:
            logging.info(
                "JSONL for sample %i could not be loaded at position %i: %s\n%s",
                index,
                position,
                str(e),
                line,
            )
            print(
                f"JSONL for sample {index} could not be loaded at position {position}: {str(e)}\n {line}",
                index,
                position,
                str(e),
                line,
            )
            return self.empty_sample
        img_path: Path = self.path_to_root / self.root_name / data.pop("image")
        if not img_path.exists():
            logging.info("Sample %s could not be found.", img_path)
            print("Sample %s could not be found.", img_path)
            return self.empty_sample
        try:
            img = Image.open(img_path)
        except UnidentifiedImageError:
            logging.info("Image %s could not be opened.", img_path)
            print("Image %s could not be opened.", img_path)
            return self.empty_sample
        return {"image": img, "ground_truth": data.pop("markdown"), "meta": data}

    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]


class NougatDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        root_name: str = "",
        processor = NougatProcessor.from_pretrained(cfg.base_model),
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id

        template = "%s"
        self.dataset = SciPDFDataset(dataset_path, split=self.split, template=template, root_name=root_name)
        self.dataset_length = len(self.dataset)
        self.processor = processor

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        # If sample is None, print error message and return None
        if sample is None:
            print(f"Sample {idx} is None")
            return None
        # Sample example: {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=816x1056 at 0x7817FC515EB0>, 'ground_truth': 'sin fulla utveckling...', 'meta': {'meta': '[]'}}
        # display(sample["image"])

        # Inputs
        input_tensor = self.processor(sample["image"], return_tensors="pt").pixel_values
        input_tensor = input_tensor.squeeze()

        # Targets
        target_sequence = sample["ground_truth"]
        target_sequence = unicodedata.normalize("NFC", target_sequence)
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        # labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # Model doesn't need to predict pad token
        # Maybe add attention mask stuff...

        return input_tensor, labels, target_sequence
