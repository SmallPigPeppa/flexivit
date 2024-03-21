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
import torch


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str,
            num_classes: int,
            image_size: int = 224,
            patch_size: int = 16,
            resize_type: str = "pi",
            ckpt_path: str = None,
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
        self.ckpt_path = ckpt_path

        # Load original weights
        print(f"Loading weights {self.weights}")
        if self.ckpt_path is not None:
            self.net = create_model(self.weights, pretrained=True)
            # self.net = create_model(self.weights, pretrained=False,
            #                         checkpoint_path=self.ckpt_path)
            # self.net = create_model(self.weights, pretrained=False)
            # model_path = self.ckpt_path
            # self.net.load_state_dict(torch.load(model_path))
        else:
            orig_net = create_model(self.weights, pretrained=True)

        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def test_step(self, batch, _):
        x, y = batch

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        acc = self.acc(pred, y)

        preds = torch.argmax(pred, dim=1)
        acc2 = torch.sum(preds == y) / y.shape[0]

        # Log
        self.log(f"test_loss", loss)
        self.log(f"test_acc", acc)
        self.log(f"test_acc2", acc2)

        return {"test_acc": acc}

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
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts

    dm = DataModule(**args["data"])
    # args["model"]["n_classes"] = dm.num_classes
    # args["model"]["image_size"] = dm.size
    model = ClassificationEvaluator(**args["model"])
    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(name='test', project='flexivit', entity='pigpeppa', offline=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)

    trainer.test(model, datamodule=dm)
