from typing import Callable, Optional, Sequence, Union
import pytorch_lightning as pl
from timm.data import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
                       OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
from timm.data.transforms_factory import transforms_imagenet_eval
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        is_lmdb: bool = False,
        root: str = "data/",
        num_classes: int = 1000,
        size: int = 224,
        crop_pct: float = 1.0,
        interpolation: str = "bicubic",
        mean: Union[Sequence[float], str] = (0.485, 0.456, 0.406),
        std: Union[Sequence[float], str] = (0.229, 0.224, 0.225),
        batch_size: int = 256,
        workers: int = 4,
    ):
        """Classification Evaluation Datamodule

        Args:
            is_lmdb: Whether the dataset is an lmdb file
            root: Path to dataset directory or lmdb file
            num_classes: Number of target classes
            size: Input image size
            crop_pct: Center crop percentage
            mean: Normalization means. Can be 'clip' or 'imagenet' to use the respective defaults
            std: Normalization standard deviations. Can be 'clip' or 'imagenet' to use the respective defaults
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.is_lmdb = is_lmdb
        self.root = root
        self.num_classes = num_classes
        self.size = size
        self.crop_pct = crop_pct
        self.interpolation = interpolation
        self.batch_size = batch_size
        self.workers = workers

        if mean == "clip":
            self.mean = OPENAI_CLIP_MEAN
        elif mean == "imagenet":
            self.mean = IMAGENET_DEFAULT_MEAN
        else:
            self.mean = mean

        if std == "clip":
            self.std = OPENAI_CLIP_STD
        elif std == "imagenet":
            self.std = IMAGENET_DEFAULT_STD
        else:
            self.std = std


        self.transforms = transforms_imagenet_eval(
            img_size=self.size,
            crop_pct=self.crop_pct,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
        )

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage="test"):
        self.test_dataset = ImageFolder(root=self.root, transform=self.transform)
        print(f"Using dataset from {self.root}")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


