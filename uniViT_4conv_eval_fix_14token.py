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
from flexivit_pytorch.utils import resize_abs_pos_embed
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from timm.models._manipulate import checkpoint_seq

from timm.layers import PatchEmbed
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


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
        # self.net = create_model(self.weights, pretrained=True)
        state_dict = orig_net.state_dict()
        self.origin_state_dict = state_dict
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(
            img_size=224,
            patch_size=16,
            num_classes=self.num_classes,
            dynamic_img_size=True
        ).to(self.device)
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_0 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_1 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_2 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_3 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

        # modified
        self.modified(new_image_size=self.image_size, new_patch_size=self.patch_size)

    # def forward(self, x):
    #     return self.net(x)
    #
    # def training_step_old(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = self.loss_fn(logits, y)
    #     acc = self.acc(logits, y)
    #
    #     # out dict
    #     out_dict = {'loss': loss, 'train_loss': loss, 'train_acc': acc}
    #     # Log
    #     self.log_dict(out_dict, on_step=False, sync_dist=True, on_epoch=True)
    #     return out_dict
    #
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits_0, logits_1, logits_2 = self.ms_forward(x)
    #     loss_0 = self.loss_fn(logits_0, y)
    #     acc_0 = self.acc_0(logits_0, y)
    #     loss_1 = self.loss_fn(logits_1, y)
    #     acc_1 = self.acc_1(logits_1, y)
    #     loss_2 = self.loss_fn(logits_2, y)
    #     acc_2 = self.acc_2(logits_2, y)
    #
    #     loss = loss_0 + loss_1 + loss_2
    #
    #     # out dict
    #     # out_dict = {'loss': loss, 'train_loss': loss, 'train_acc': acc}
    #     out_dict = {'loss': loss,
    #                 'train_loss_0': loss_0, 'train_loss_1': loss_1, 'train_loss_2': loss_2,
    #                 'train_acc_0': acc_0, 'train_acc_1': acc_1, 'train_acc_2': acc_2
    #                 }
    #     # Log
    #     self.log_dict(out_dict, on_step=False, sync_dist=True, on_epoch=True)
    #     return out_dict
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits_0, logits_1, logits_2 = self.ms_forward(x)
    #     loss_0 = self.loss_fn(logits_0, y)
    #     acc_0 = self.acc_0(logits_0, y)
    #     loss_1 = self.loss_fn(logits_1, y)
    #     acc_1 = self.acc_1(logits_1, y)
    #     loss_2 = self.loss_fn(logits_2, y)
    #     acc_2 = self.acc_2(logits_2, y)
    #
    #     loss = loss_0 + loss_1 + loss_2
    #
    #     # out dict
    #     # out_dict = {'loss': loss, 'train_loss': loss, 'train_acc': acc}
    #     out_dict = {'loss': loss,
    #                 'val_loss_0': loss_0, 'val_loss_1': loss_1, 'val_loss_2': loss_2,
    #                 'val_acc_0': acc_0, 'val_acc_1': acc_1, 'val_acc_2': acc_2
    #                 }
    #     # Log
    #     self.log_dict(out_dict, on_step=False, sync_dist=True, on_epoch=True)
    #     return out_dict
    #
    # def test_step_old(self, batch, _):
    #     x, y = batch
    #     x = F.interpolate(x, size=self.image_size, mode='bilinear')
    #
    #     # Pass through network
    #     # pred = self(x)
    #     _, _, pred = self.ms_forward(x)
    #     loss = self.loss_fn(pred, y)
    #
    #     # Get accuracy
    #     acc = self.acc(pred, y)
    #
    #     # Log
    #     self.log_dict({'test_loss': loss, 'test_acc': acc}, sync_dist=True, on_epoch=True)
    #
    #     return loss

    def test_step(self, batch, _):
        x, y = batch
        x = F.interpolate(x, size=self.image_size, mode='bilinear')

        # Pass through network
        # pred = self(x)
        pred0, pred1, pred2, pred3 = self.ms_forward(x)

        # Get accuracy
        acc_0 = self.acc_0(pred0, y)
        acc_1 = self.acc_1(pred1, y)
        acc_2 = self.acc_2(pred2, y)
        acc_3 = self.acc_3(pred3, y)

        # Log
        out_dict = {'test_acc_0': acc_0, 'test_acc_1': acc_1, 'test_acc_2': acc_2, 'test_acc_3': acc_3}
        self.log_dict(out_dict, sync_dist=True, on_epoch=True)

        return out_dict

    # def test_epoch_end_old_old(self, outputs):
    #     if self.results_path:
    #         acc = self.acc.compute().detach().cpu().item()
    #         acc = acc * 100
    #         # 让所有进程都执行到这里，但只有主进程进行写入操作
    #         if self.trainer.is_global_zero:
    #             column_name = f"{self.image_size}_{self.patch_size}"
    #
    #             if os.path.exists(self.results_path):
    #                 # 结果文件已存在，读取现有数据
    #                 results_df = pd.read_csv(self.results_path, index_col=0)
    #                 # 检查列是否存在，若不存在则添加
    #                 results_df[column_name] = acc
    #             else:
    #                 # 结果文件不存在，创建新的DataFrame
    #                 results_df = pd.DataFrame({column_name: [acc]})
    #                 # 确保目录存在
    #                 os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
    #
    #             # 保存更新后的结果
    #             results_df.to_csv(self.results_path)

    def test_epoch_end(self, outputs):
        if self.results_path:
            # 计算每个acc并乘以100
            acc_0 = self.acc_0.compute().detach().cpu().item() * 100
            acc_1 = self.acc_1.compute().detach().cpu().item() * 100
            acc_2 = self.acc_2.compute().detach().cpu().item() * 100
            acc_3 = self.acc_3.compute().detach().cpu().item() * 100
            max_acc = max(acc_0, acc_1, acc_2, acc_3)

            # 确保所有进程都执行到这里，但只有主进程进行写入操作
            if self.trainer.is_global_zero:
                column_name = f"{self.image_size}_{self.patch_size}"

                if os.path.exists(self.results_path):
                    # 结果文件已存在，读取现有数据
                    results_df = pd.read_csv(self.results_path, index_col=0)
                    # 检查列是否存在，若不存在则在适当位置添加
                    if column_name not in results_df:
                        results_df[column_name] = [None] * len(results_df)  # 先添加空列，防止DataFrame对齐问题
                else:
                    # 结果文件不存在，创建新的DataFrame，此时有5行
                    results_df = pd.DataFrame(columns=[column_name], index=['acc0', 'acc1', 'acc2', 'acc3', 'max_acc'])
                    # 确保目录存在
                    os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

                # 更新DataFrame中的值
                results_df.at['acc0', column_name] = acc_0
                results_df.at['acc1', column_name] = acc_1
                results_df.at['acc2', column_name] = acc_2
                results_df.at['acc3', column_name] = acc_3
                results_df.at['max_acc', column_name] = max_acc

                # 保存更新后的结果
                results_df.to_csv(self.results_path)

    # def configure_optimizers(self):
    #     self.lr = 0.002
    #     self.wd = 5e-4
    #     self.max_epochs = self.trainer.max_epochs
    #
    #     # params_to_optimize = list(self.patch_embed_56.parameters()) + \
    #     #                      list(self.patch_embed_112.parameters()) + \
    #     #                      list(self.patch_embed_224.parameters())
    #
    #     params_to_optimize = list(self.patch_embed_56.parameters()) + \
    #                          list(self.patch_embed_112.parameters())
    #     # list(self.patch_embed_224.parameters())
    #
    #     optimizer = torch.optim.SGD(
    #         params_to_optimize,
    #         lr=self.lr,
    #         weight_decay=self.wd,
    #         momentum=0.9)
    #
    #     # optimizer = torch.optim.SGD(
    #     #     self.parameters(),
    #     #     lr=self.lr,
    #     #     weight_decay=self.wd,
    #     #     momentum=0.9)
    #
    #     scheduler = LinearWarmupCosineAnnealingLR(
    #         optimizer,
    #         warmup_epochs=5,
    #         max_epochs=self.max_epochs,
    #         warmup_start_lr=0.01 * self.lr,
    #         eta_min=0.01 * self.lr,
    #     )
    #     return [optimizer], [scheduler]

    # return [optimizer]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.patch_embed(x)
        x = self.net._pos_embed(x)
        x = self.net.patch_drop(x)
        x = self.net.norm_pre(x)
        if self.net.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.net.blocks, x)
        else:
            x = self.net.blocks(x)
        x = self.net.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.net.attn_pool is not None:
            x = self.net.attn_pool(x)
        elif self.net.global_pool == 'avg':
            x = x[:, self.net.num_prefix_tokens:].mean(dim=1)
        elif self.net.global_pool:
            x = x[:, 0]  # class token
        x = self.net.fc_norm(x)
        x = self.net.head_drop(x)
        return x if pre_logits else self.net.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    # def ms_forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x_0 = F.interpolate(x, size=56, mode='bilinear')
    #     x_0 = self.patch_embed_56(x_0)
    #
    #     x_1 = F.interpolate(x, size=112, mode='bilinear')
    #     x_1 = self.patch_embed_112(x_1)
    #
    #     x_2 = F.interpolate(x, size=224, mode='bilinear')
    #     x_2 = self.patch_embed_224(x_2)
    #
    #     return self(x_0), self(x_1), self(x_2)

    def ms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_0 = F.interpolate(x, size=56, mode='bilinear')
        x_0 = self.patch_embed_4x4(x, patch_size=self.patch_size)

        # x_1 = F.interpolate(x, size=112, mode='bilinear')
        x_1 = self.patch_embed_8x8(x, patch_size=self.patch_size)

        # x_2 = F.interpolate(x, size=224, mode='bilinear')
        x_2 = self.patch_embed_12x12(x, patch_size=self.patch_size)

        x_3 = self.patch_embed_16x16(x, patch_size=self.patch_size)

        return self(x_0), self(x_1), self(x_2), self(x_3)

    def modified(self, new_image_size=224, new_patch_size=16):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            # flatten deferred until after pos embed
            self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed_4x4 = self.get_new_patch_embed(new_image_size=56, new_patch_size=4)
        self.patch_embed_8x8 = self.get_new_patch_embed(new_image_size=112, new_patch_size=8)
        self.patch_embed_12x12 = self.get_new_patch_embed(new_image_size=192, new_patch_size=12)
        self.patch_embed_16x16 = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        self.patch_embed_16x16_origin = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        # import pdb;pdb.set_trace()

        self.net.patch_embed = nn.Identity()

    def get_new_patch_embed(self, new_image_size, new_patch_size):
        new_patch_embed = FlexiPatchEmbed(
            img_size=new_image_size,
            patch_size=new_patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            # dynamic_img_pad=self.dynamic_img_pad,
            **self.embed_args,
        )
        if hasattr(self.net.patch_embed.proj, 'weight'):
            origin_weight = self.net.patch_embed.proj.weight.clone().detach()
            # new_weight = pi_resize_patch_embed(
            #     patch_embed=self.origin_state_dict["patch_embed.proj.weight"],
            #     new_patch_size=(new_patch_size, new_patch_size)
            # )
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight, new_patch_size=(new_patch_size, new_patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            # new_patch_embed.proj.bias = nn.Parameter(torch.tensor(self.origin_state_dict["patch_embed.proj.bias"]),
            #                                          requires_grad=True)
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
    # wandb_logger = WandbLogger(name='ft-part-conv-uniViT', project='uniViT', entity='pigpeppa', offline=False)
    # checkpoint_callback = ModelCheckpoint(monitor="val_acc_2", mode="max", dirpath='ckpt/uniViT/3_add_random_resize-v2', save_top_k=1,save_last=True)
    # trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])

    trainer = pl.Trainer.from_argparse_args(args)
    # for image_size, patch_size in [(32, 4), (48, 4), (64, 4), (80, 8), (96, 8), (112, 8), (128, 8), (144, 16),
    #                                (160, 16), (176, 16), (192, 16), (208, 16), (224, 16)]:
    for image_size, patch_size in [(28, 2), (42, 3), (56, 4), (70, 5), (84, 6), (98, 7), (112, 8), (126, 9), (140, 10),
                                   (154, 11), (168, 12),
                                   (182, 13), (196, 14), (210, 15), (224, 16), (238, 17), (252, 18)]:
        # for image_size, patch_size in [(224, 16)]:
        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        # model = ClassificationEvaluator(**args["model"])
        # ckpt_path = '/mnt/mmtech01/usr/liuwenzhuo/code/test-code/flexivit/ckpt/uniViT/3_add_random_resize-v2/last-v1.ckpt'
        ckpt_path = '/mnt/mmtech01/usr/liuwenzhuo/code/test-code/flexivit/ckpt/uniViT/3_add_random_resize2-v3/last.ckpt'
        ckpt_path = '/mnt/mmtech01/usr/liuwenzhuo/code/test-code/uniViT/ckpt/uniViT/no-random-resize-4conv/last.ckpt'
        model = ClassificationEvaluator.load_from_checkpoint(checkpoint_path=ckpt_path, strict=True, **args["model"])
        data_config = timm.data.resolve_model_data_config(model.net)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                                shuffle=False, pin_memory=True)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        train_dataset = ImageFolder(root=os.path.join(args.root, 'train'), transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.works,
                                  shuffle=True, pin_memory=True)
        # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model, dataloaders=val_loader)
