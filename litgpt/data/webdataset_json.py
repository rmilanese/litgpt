# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union, List

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from litgpt.prompts import PromptStyle
from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.tokenizer import Tokenizer

# Optionally set WebDataset-related environment variables for verbosity/control
os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"

@dataclass
class WebDatasetJSON(DataModule):
    """Loads JSON samples from a WebDataset tar archive quickly.
    
    The tar shards are assumed to contain JSON files (with a '.json' extension).
    Each sample is streamed and decoded on the fly, then converted to a list for
    supervised fine-tuning via the SFTDataset interface.
    
    Example URL format (brace expansion supported):
      "https://storage.googleapis.com/mybucket/mydata-{000000..000009}.tar"
    """
    url: str
    """URL (or local path pattern) to the WebDataset tar shards."""
    shuffle_buffer: int = 1000
    """Buffer size for shuffling samples within each shard."""
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (using ignore_index)."""
    val_split_fraction: Optional[float] = 0.1
    """Fraction of samples to use for validation (the rest for training)."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """Prompt style to apply to instruction prompts."""
    ignore_index: int = -100
    """Index to use for ignored tokens in the labels."""
    seed: int = 42
    """Random seed for shuffling and train/validation splitting."""
    num_workers: int = 4
    """Number of DataLoader workers for loading and processing."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:
        # Import WebDataset only when needed
        import webdataset as wds

        # Create the WebDataset pipeline:
        # - Shuffle shards using the specified buffer
        # - Decode raw bytes as UTF-8 text (each sample is assumed to be a JSON file)
        # - Group the sample into a tuple keyed by "json"
        dataset = (
            wds.WebDataset(self.url, cache_dir="/tmp/webdataset_cache")
            .shuffle(self.shuffle_buffer)
            .decode("utf8")
            .to_tuple("json")
        )

        # Convert the tuple (json_text,) to a dict by parsing the JSON
        def parse_json(sample: Tuple[str]) -> Any:
            json_text, = sample
            return json.loads(json_text)

        dataset = dataset.map(parse_json)

        # Convert the WebDataset iterator into a list (assumes dataset size is manageable)
        data: List[Any] = list(dataset)

        # Split the data into training and validation sets
        if self.val_split_fraction is None:
            raise ValueError("val_split_fraction must be provided for splitting the data.")
        split_idx = int((1 - self.val_split_fraction) * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )
