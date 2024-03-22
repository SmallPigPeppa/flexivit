import os
from typing import Callable, Optional, Sequence, Union
import pandas as pd
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
from data_utils.imagenet_val import DataModule
import torch.nn.functional as F
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
from flexivit_pytorch.utils import resize_abs_pos_embed


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str,
            num_classes: int,
            image_size: int = 224,
            patch_size: int = 16,
            resize_type: str = "pi",
            results_path: Optional[str] = None,
    ):
        """Classification Evaluator

        Args:
            weights: Name of model weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resize_type: Patch embed resize method. One of ["pi", "interpolate"]
            results_path: Path to write evaluation results. Does not write results if empty
        """
        super().__init__()
        self.save_hyperparameters()
        self.weights = weights
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.resize_type = resize_type
        self.results_path = results_path

        # Load original weights
        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        state_dict = orig_net.state_dict()

        # Adjust patch embedding
        if self.resize_type == "pi":
            state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        elif self.resize_type == "interpolate":
            state_dict["patch_embed.proj.weight"] = interpolate_resize_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        else:
            raise ValueError(
                f"{self.resize_type} is not a valid value for --model.resize_type. Should be one of ['flexi', 'interpolate']"
            )

        # Adjust position embedding
        if "pos_embed" in state_dict.keys():
            grid_size = self.image_size // self.patch_size
            state_dict["pos_embed"] = resize_abs_pos_embed(
                state_dict["pos_embed"], new_size=(grid_size, grid_size)
            )

        # Load adjusted weights into model with target patch and image sizes
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(
            img_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
        )
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def test_step(self, batch, _):
        x, y = batch
        x = F.interpolate(x, size=self.image_size, mode='bilinear')

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        acc = self.acc(pred, y)

        # Log
        self.log(f"test_loss", loss)
        self.log(f"test_acc", acc)

        return loss

    def test_epoch_end(self, _):
        if self.results_path:
            acc = self.acc.compute().detach().cpu().item()
            results = pd.DataFrame(
                {
                    "model": [self.weights],
                    "acc": [round(acc, 4)],
                    "patch_size": [self.patch_size],
                    "image_size": [self.image_size],
                    "resize_type": [self.resize_type],
                }
            )

            if not os.path.exists(os.path.dirname(self.results_path)):
                os.makedirs(os.path.dirname(self.results_path))

            results.to_csv(
                self.results_path,
                mode="a",
                header=not os.path.exists(self.results_path),
            )


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(DataModule, "data")
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    parser.link_arguments("data.num_classes", "model.num_classes")
    parser.link_arguments("data.size", "model.image_size")
    # parser.link_arguments("max_epochs", "model.max_epochs")
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts

    # wandb_logger = WandbLogger(name='test', project='flexivit', entity='pigpeppa', offline=False)
    # trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer = pl.Trainer.from_argparse_args(args)
    dm = DataModule(**args["data"])


    # for image_size, patch_size in [(32, 4), (48, 4), (64, 4), (80, 8), (96, 8), (112, 8), (128, 8), (144, 16),
    #                                (160, 16), (176, 16), (192, 16), (208, 16), (224, 16)]:
    for image_size, patch_size in [(32, 4), (56, 4), (64, 8), (112, 8), (224, 16)]:
        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        model = ClassificationEvaluator(**args["model"])
        trainer.test(model, datamodule=dm)
