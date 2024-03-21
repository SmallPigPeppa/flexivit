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
import torch.optim as optim
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


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
            max_epochs: int = 20,
            lr: float = 0.1,
            wd: float = 5e-4,
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
        self.max_epochs = max_epochs
        self.lr = lr
        self.wd = wd

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

        # 冻结除FC层之外的所有层
        for param in self.net.parameters():
            param.requires_grad = False
        # 假设全连接层的名称是 'head'，这在不同的模型中可能有所不同
        for param in self.net.head.parameters():
            param.requires_grad = True

    # def configure_optimizers(self):
    #     # 只优化全连接层的参数
    #     optimizer = optim.Adam(self.net.head.parameters(), lr=1e-3)
    #     return optimizer

    def configure_optimizers(self):
        # self.lr=0.1
        # self.wd=5e-4

        optimizer = torch.optim.SGD(
            self.net.head.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            momentum=0.9)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.max_epochs,
            warmup_start_lr=0.01 * self.learning_rate,
            eta_min=0.01 * self.learning_rate,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

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
    parser.link_arguments("max_epochs", "model.max_epochs")
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts

    dm = DataModule(**args["data"])
    from data_utils.imagenet_dali import ClassificationDALIDataModule

    dm_dali = ClassificationDALIDataModule(
        train_data_path=os.path.join(args["data"].root, 'train'),
        val_data_path=os.path.join(args["data"].root, 'val'),
        num_workers=args["data"].workers,
        batch_size=args["data"].batch_size)
    # args["model"]["n_classes"] = dm.num_classes
    # args["model"]["image_size"] = dm.size
    model = ClassificationEvaluator(**args["model"])
    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(name='test', project='flexivit', entity='pigpeppa', offline=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)

    trainer.fit(model, dm_dali)
    trainer.test(model, datamodule=dm)
    # model.eval()
    # trainer.test(model, datamodule=dm)
