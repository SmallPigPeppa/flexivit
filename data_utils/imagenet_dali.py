import math
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import pytorch_lightning as pl
import torch
import torch.nn as nn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class BaseWrapper(DALIGenericIterator):
    """Temporary fix to handle LastBatchPolicy.DROP."""

    def __len__(self):
        size = (
            self._size_no_pad // self._shards_num
            if self._last_batch_policy == LastBatchPolicy.DROP
            else self.size
        )
        if self._reader_name:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / self.batch_size)

            return size // self.batch_size
        else:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / (self._devices * self.batch_size))

            return size // (self._devices * self.batch_size)


class Wrapper(BaseWrapper):
    def __next__(self):
        batch = super().__next__()
        x, target = batch[0]["x"], batch[0]["label"]
        target = target.squeeze(-1).long()
        # PyTorch Lightning does double buffering
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1316,
        # and as DALI owns the tensors it returns the content of it is trashed so the copy needs,
        # to be made before returning.
        x = x.detach().clone()
        target = target.detach().clone()
        return x, target


class ClassificationDALIDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_path: Union[str, Path],
            val_data_path: Union[str, Path],
            batch_size: int,
            num_workers: int = 4,
            data_fraction: float = -1.0,
            dali_device: str = "gpu",
    ):
        """DataModule for classification data using Nvidia DALI.

        Args:
            dataset (str): dataset name.
            train_data_path (Union[str, Path]): path where the training data is located.
            val_data_path (Union[str, Path]): path where the validation data is located.
            batch_size (int): batch size..
            num_workers (int, optional): number of parallel workers. Defaults to 4.
            data_fraction (float, optional): percentage of data to use.
                Use all data when set to -1.0. Defaults to -1.0.
            dali_device (str, optional): device used by the dali pipeline.
                Either 'gpu' or 'cpu'. Defaults to 'gpu'.
        """

        super().__init__()

        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)

        self.num_workers = num_workers

        self.batch_size = batch_size

        self.data_fraction = data_fraction

        self.dali_device = dali_device
        assert dali_device in ["gpu", "cpu"]

        # handle custom data by creating the needed pipeline
        self.pipeline_class = NormalPipelineBuilder

    def setup(self, stage: Optional[str] = None):
        # extra info about training
        self.device_id = self.trainer.local_rank
        self.shard_id = self.trainer.global_rank
        self.num_shards = self.trainer.world_size

        # get current device
        if torch.cuda.is_available() and self.dali_device == "gpu":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

    def train_dataloader(self):
        train_pipeline_builder = self.pipeline_class(
            self.train_data_path,
            validation=False,
            batch_size=self.batch_size,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
            data_fraction=self.data_fraction,
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        train_loader = Wrapper(
            train_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )

        self.dali_epoch_size = train_pipeline.epoch_size("Reader")

        return train_loader

    def val_dataloader(self) -> DALIGenericIterator:
        val_pipeline_builder = self.pipeline_class(
            self.val_data_path,
            validation=True,
            batch_size=self.batch_size,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
        )
        val_pipeline = val_pipeline_builder.pipeline(
            batch_size=val_pipeline_builder.batch_size,
            num_threads=val_pipeline_builder.num_threads,
            device_id=val_pipeline_builder.device_id,
            seed=val_pipeline_builder.seed,
        )
        val_pipeline.build()

        val_loader = Wrapper(
            val_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        return val_loader


class NormalPipelineBuilder:
    def __init__(
            self,
            data_path: str,
            batch_size: int,
            device: str,
            validation: bool = False,
            device_id: int = 0,
            shard_id: int = 0,
            num_shards: int = 1,
            num_threads: int = 4,
            seed: int = 12,
            data_fraction: float = -1.0,
    ):
        """Initializes the pipeline for validation or linear eval training.

        If validation is set to True then images will only be resized to 256px and center cropped
        to 224px, otherwise random resized crop, horizontal flip are applied. In both cases images
        are normalized.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            validation (bool): whether it is validation or training. Defaults to False. Defaults to
                False.
            device_id (int): id of the device used to initialize the seed and for parent class.
                Defaults to 0.
            shard_id (int): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int): total number of shards. Defaults to 1.
            num_threads (int): number of threads to run in parallel. Defaults to 4.
            seed (int): seed for random number generation. Defaults to 12.
            data_fraction (float): percentage of data to use. Use all data when set to -1.0.
                Defaults to -1.0.
        """

        super().__init__()

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed + device_id

        self.device = device
        self.validation = validation

        # manually load files and labels
        labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())
        data = [
            (data_path / label / file, label_idx)
            for label_idx, label in enumerate(labels)
            for file in sorted(os.listdir(data_path / label))
        ]
        files, labels = map(list, zip(*data))

        # sample data if needed
        if data_fraction > 0:
            assert data_fraction < 1, "data_fraction must be smaller than 1."

            from sklearn.model_selection import train_test_split

            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )

        self.reader = ops.readers.File(
            files=files,
            labels=labels,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=not self.validation,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        # crop operations
        if self.validation:
            self.resize = ops.Resize(
                device=self.device,
                resize_shorter=256,
                interp_type=types.INTERP_CUBIC,
            )
            # center crop and normalize
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )
        else:
            self.resize = ops.RandomResizedCrop(
                device=self.device,
                size=224,
                random_area=(0.08, 1.0),
                interp_type=types.INTERP_CUBIC,
            )
            # normalize and horizontal flip
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )

        self.coin05 = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    @pipeline_def
    def pipeline(self):
        """Defines the computational pipeline for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        images = self.resize(images)

        if self.validation:
            # crop and normalize
            images = self.cmn(images)
        else:
            # normalize and maybe apply horizontal flip with 0.5 chance
            images = self.cmn(images, mirror=self.coin05())

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (images, labels)


if __name__ == '__main__':
    dali_datamodule = ClassificationDALIDataModule(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        data_fraction=args.data_fraction,
        dali_device='gpu',
    )

    # use normal torchvision dataloader for validation to save memory
    dali_datamodule.val_dataloader = lambda: val_loader
