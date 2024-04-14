import os
from typing import Callable, Optional, Sequence, Union
import pandas as pd
import torch
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn as nn
import random
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from flexivit_pytorch.myflex import FlexiOverlapPatchEmbed_DB as FlexiOverlapPatchEmbed


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
        self.net = create_model(weights, pretrained=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

        # modified
        self.modified()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits_4x4, logits_8x8, logits_12x12, logits_16x16 = self.rand_ms_forward(x)
        loss_4x4 = self.loss_fn(logits_4x4, y)
        acc_4x4 = self.acc(logits_4x4, y)
        loss_8x8 = self.loss_fn(logits_8x8, y)
        acc_8x8 = self.acc(logits_8x8, y)
        loss_12x12 = self.loss_fn(logits_12x12, y)
        acc_12x12 = self.acc(logits_12x12, y)
        loss_16x16 = self.loss_fn(logits_16x16, y)
        acc_16x16 = self.acc(logits_16x16, y)

        loss = loss_4x4 + loss_8x8 + loss_12x12 + loss_16x16
        out_dict = {'loss': loss,
                    'train_loss_4x4': loss_4x4,
                    'train_loss_8x8': loss_8x8,
                    'train_loss_12x12': loss_12x12,
                    'train_loss_16x16': loss_16x16,
                    'train_acc_4x4': acc_4x4,
                    'train_acc_8x8': acc_8x8,
                    'train_acc_12x12': acc_12x12,
                    'train_acc_16x16': acc_16x16
                    }
        # Log
        self.log_dict(out_dict, on_step=False, sync_dist=True, on_epoch=True)
        return out_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_4x4, logits_8x8, logits_12x12, logits_16x16 = self.ms_forward(x)
        loss_4x4 = self.loss_fn(logits_4x4, y)
        acc_4x4 = self.acc(logits_4x4, y)
        loss_8x8 = self.loss_fn(logits_8x8, y)
        acc_8x8 = self.acc(logits_8x8, y)
        loss_12x12 = self.loss_fn(logits_12x12, y)
        acc_12x12 = self.acc(logits_12x12, y)
        loss_16x16 = self.loss_fn(logits_16x16, y)
        acc_16x16 = self.acc(logits_16x16, y)

        loss = loss_4x4 + loss_8x8 + loss_12x12 + loss_16x16
        out_dict = {'val_loss': loss,
                    'val_loss_4x4': loss_4x4,
                    'val_loss_8x8': loss_8x8,
                    'val_loss_12x12': loss_12x12,
                    'val_loss_16x16': loss_16x16,
                    'val_acc_4x4': acc_4x4,
                    'val_acc_8x8': acc_8x8,
                    'val_acc_12x12': acc_12x12,
                    'val_acc_16x16': acc_16x16
                    }
        self.log_dict(out_dict, on_step=False, sync_dist=True, on_epoch=True)
        return out_dict

    def test_step(self, batch, _):
        x, y = batch
        x = F.interpolate(x, size=self.image_size, mode='bilinear')

        # Pass through network
        # pred = self(x)
        _, _, _, pred = self.ms_forward(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        acc = self.acc(pred, y)

        # Log
        self.log_dict({'test_loss': loss, 'test_acc': acc}, sync_dist=True, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs):
        if self.results_path:
            acc = self.acc.compute().detach().cpu().item()
            acc = acc * 100
            # 让所有进程都执行到这里，但只有主进程进行写入操作
            if self.trainer.is_global_zero:
                column_name = f"{self.image_size}_{self.patch_size}"

                if os.path.exists(self.results_path):
                    # 结果文件已存在，读取现有数据
                    results_df = pd.read_csv(self.results_path, index_col=0)
                    # 检查列是否存在，若不存在则添加
                    results_df[column_name] = acc
                else:
                    # 结果文件不存在，创建新的DataFrame
                    results_df = pd.DataFrame({column_name: [acc]})
                    # 确保目录存在
                    os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

                # 保存更新后的结果
                results_df.to_csv(self.results_path)

    def configure_optimizers(self):
        self.lr = 0.001
        self.wd = 5e-4
        self.max_epochs = self.trainer.max_epochs

        params_to_optimize = list(self.patch_embed_3x3_s1.parameters()) + \
                             list(self.patch_embed_5x5_s2.parameters()) + \
                             list(self.patch_embed_7x7_s3.parameters()) + \
                             list(self.patch_embed_7x7_s4.parameters())

        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=self.lr,
            weight_decay=self.wd,
            momentum=0.9)

        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.wd,
        #     momentum=0.9)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.max_epochs,
            warmup_start_lr=0.01 * self.lr,
            eta_min=0.01 * self.lr,
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.net.forward_head(x)
        return x

    # def forward_after_patch_embed(self, x):
    #     x = self.net.stages(x)
    #     x = self.net.forward_head(x)
    #     return x

    def forward_after_patch_embed(self, x):
        x = self.net.patch_embed.norm(x)
        x = self.net.stages(x)
        x = self.net.forward_head(x)
        return x

    def ms_forward(self, x):
        x_3x3 = F.interpolate(x, size=56, mode='bilinear')
        x_3x3 = self.patch_embed_3x3_s1(x_3x3, patch_size=3, stride=1)

        x_5x5 = F.interpolate(x, size=112, mode='bilinear')
        x_5x5 = self.patch_embed_5x5_s2(x_5x5, patch_size=5, stride=2)

        x_7x7 = F.interpolate(x, size=168, mode='bilinear')
        x_7x7 = self.patch_embed_7x7_s3(x_7x7, patch_size=7, stride=3)

        x_7x7_s4 = F.interpolate(x, size=224, mode='bilinear')
        x_7x7_s4 = self.patch_embed_7x7_s4(x_7x7_s4, patch_size=7, stride=4)

        return self.forward_after_patch_embed(x_3x3), \
            self.forward_after_patch_embed(x_5x5), \
            self.forward_after_patch_embed(x_7x7), \
            self.forward_after_patch_embed(x_7x7_s4)


    def rand_ms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # 随机选择imagesize
        img_size_3x3 = random.choice([28, 42, 56, 70, 84])
        x_3x3 = F.interpolate(x, size=(img_size_3x3, img_size_3x3), mode='bilinear')
        x_3x3 = self.patch_embed_3x3_s1(x_3x3, patch_size=3, stride=1)

        img_size_5x5 = random.choice([84, 98, 112, 126, 140])
        x_5x5 = F.interpolate(x, size=(img_size_5x5, img_size_5x5), mode='bilinear')
        x_5x5 = self.patch_embed_5x5_s2(x_5x5, patch_size=5, stride=2)

        img_size_7x7 = random.choice([140, 154, 168, 182, 196])
        x_7x7 = F.interpolate(x, size=(img_size_7x7, img_size_7x7), mode='bilinear')
        x_7x7 = self.patch_embed_7x7_s3(x_7x7, patch_size=7, stride=3)

        img_size_7x7_s4 = random.choice([196, 210, 224, 238, 252])
        x_7x7_s4 = F.interpolate(x, size=(img_size_7x7_s4, img_size_7x7_s4), mode='bilinear')
        x_7x7_s4 = self.patch_embed_7x7_s4(x_7x7_s4, patch_size=7, stride=4)

        return self.forward_after_patch_embed(x_3x3), \
            self.forward_after_patch_embed(x_5x5), \
            self.forward_after_patch_embed(x_7x7), \
            self.forward_after_patch_embed(x_7x7_s4)

    def modified(self):
        self.in_chans = 3
        self.embed_dim = 64
        self.patch_embed_3x3_s1 = self.get_new_patch_embed(new_patch_size=3, new_stride=1)
        self.patch_embed_5x5_s2 = self.get_new_patch_embed(new_patch_size=5, new_stride=2)
        self.patch_embed_7x7_s3 = self.get_new_patch_embed(new_patch_size=7, new_stride=3)
        self.patch_embed_7x7_s4 = self.get_new_patch_embed(new_patch_size=7, new_stride=4)

    def get_new_patch_embed(self, new_patch_size, new_stride):
        new_patch_embed = FlexiOverlapPatchEmbed(
            patch_size=new_patch_size,
            stride=new_stride,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        if hasattr(self.net.patch_embed.proj, 'weight'):
            origin_weight = self.net.patch_embed.proj.weight.clone().detach()
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight, new_patch_size=(new_patch_size, new_patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(self.net.patch_embed.proj.bias.clone().detach(),
                                                     requires_grad=True)

        return new_patch_embed

if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--works", type=int, default=4)
    parser.add_argument("--root", type=str, default='./data')
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts

    wandb_logger = WandbLogger(name='add-random-resize-4conv-fix14token-2range-pvt-fixnorm', project='L2P',
                               entity='pigpeppa', offline=False)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc_16x16", mode="max",
                                          dirpath='ckpt/L2P/add_random_resize_4conv_fix14token_2range/pvt_fixnorm',
                                          save_top_k=1,
                                          save_last=True)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])
    # lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # trainer = pl.Trainer.from_argparse_args(args)

    for image_size, patch_size in [(224, 16)]:
        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        model = ClassificationEvaluator(**args["model"])
        data_config = timm.data.resolve_model_data_config(model.net)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                                shuffle=False, pin_memory=True)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        train_dataset = ImageFolder(root=os.path.join(args.root, 'train'), transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.works,
                                  shuffle=True, pin_memory=True)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # trainer.test(model, dataloaders=val_loader)
