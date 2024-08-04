import os
from typing import Callable, Optional, Sequence, Union
import pandas as pd
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
from flexivit_pytorch.utils import resize_abs_pos_embed_deit3b as resize_abs_pos_embed
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from timm.models._manipulate import checkpoint_seq
from timm.layers import resample_abs_pos_embed
from torch import nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models.flex_patch_embed import FlexiPatchEmbed


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
                state_dict["pos_embed"], new_size=(grid_size, grid_size), num_prefix_tokens=0
            )

        # Load adjusted weights into model with target patch and image sizes
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(
            img_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
        ).to(self.device)
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc0 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc1 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc2 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc3 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

        self.modified()

        # self.pos_embed_height = nn.Parameter(torch.randn(self.image_size // self.patch_size, self.net.embed_dim))
        # self.pos_embed_width = nn.Parameter(torch.randn(self.image_size // self.patch_size, self.net.embed_dim))
        # self.pos_height = self.image_size // self.patch_size
        # self.pos_width = self.image_size // self.patch_size

    # def _pos_embed_learn2D(self, x, patch_positions=None):
    #     batch_size, num_patches, _ = x.shape
    #
    #     if patch_positions is None:
    #         # Generate patch positions based on the assumed grid layout
    #         assert num_patches == self.pos_height * self.pos_width, "Number of patches does not match the specified grid dimensions"
    #         patch_positions = torch.cartesian_prod(torch.arange(self.pos_height), torch.arange(self.pos_width))
    #         patch_positions = patch_positions.repeat(batch_size, 1, 1).view(batch_size, self.pos_height, self.pos_width,
    #                                                                         2).to(
    #             self.device)
    #
    #     # Unbind patch positions into height and width indices
    #     h_indices, w_indices = patch_positions.unbind(dim=-1)
    #
    #     # Retrieve positional embeddings
    #     h_pos = self.pos_embed_height[h_indices]
    #     w_pos = self.pos_embed_width[w_indices]
    #
    #     pos_embed = h_pos + w_pos
    #
    #     # Combine embeddings and positional encodings
    #     to_cat = []
    #     if self.net.cls_token is not None:
    #         to_cat.append(self.net.cls_token.expand(batch_size, -1, -1))
    #     if self.net.reg_token is not None:
    #         to_cat.append(self.net.reg_token.expand(batch_size, -1, -1))
    #
    #     no_embed_class = True
    #     # if self.net.no_embed_class:
    #     if no_embed_class:
    #         x = x + pos_embed.view(batch_size, num_patches, -1)  # Add pos_embed first
    #         if to_cat:
    #             x = torch.cat(to_cat + [x], dim=1)
    #     else:
    #         if to_cat:
    #             x = torch.cat(to_cat + [x], dim=1)
    #         x = x + pos_embed.view(batch_size, num_patches, -1)  # Add pos_embed after concat
    #
    #     return self.net.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.patch_embed(x)
        x = self.net._pos_embed(x)
        x = self.net.patch_drop(x)
        x = self.net.norm_pre(x)
        if self.net.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
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

    def ms_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_4x4 = F.interpolate(x, size=56, mode='bilinear')
        x_4x4 = self.patch_embed_4x4(x_4x4, patch_size=4)
        logits_4x4 = self.forward(x_4x4)

        x_8x8 = F.interpolate(x, size=112, mode='bilinear')
        x_8x8 = self.patch_embed_8x8(x_8x8, patch_size=8)
        logits_8x8 = self.forward(x_8x8)

        x_12x12 = F.interpolate(x, size=168, mode='bilinear')
        x_12x12 = self.patch_embed_12x12(x_12x12, patch_size=12)
        logits_12x12 = self.forward(x_12x12)

        x_16x16 = F.interpolate(x, size=224, mode='bilinear')
        x_16x16 = self.patch_embed_16x16(x_16x16, patch_size=16)
        logits_16x16 = self.forward(x_16x16)

        return logits_4x4, logits_8x8, logits_12x12, logits_16x16

    def training_step(self, batch, batch_idx):
        out_dict = self.share_step(batch, batch_idx)
        log_dict = {f'train/{key}': value for key, value in out_dict.items()}
        self.log_dict(log_dict, on_step=False, sync_dist=True, on_epoch=True)
        return out_dict['loss']

    def validation_step(self, batch, batch_idx):
        out_dict = self.share_step(batch, batch_idx)
        log_dict = {f'val/{key}': value for key, value in out_dict.items()}
        self.log_dict(log_dict, on_step=False, sync_dist=True, on_epoch=True)
        return out_dict['loss']

    def share_step(self, batch, batch_idx):
        x, y = batch
        logits_4x4, logits_8x8, logits_12x12, logits_16x16 = self.ms_forward(x)
        loss_4x4 = self.loss_fn(logits_4x4, y)
        acc_4x4 = self.acc0(logits_4x4, y)
        loss_8x8 = self.loss_fn(logits_8x8, y)
        acc_8x8 = self.acc1(logits_8x8, y)
        loss_12x12 = self.loss_fn(logits_12x12, y)
        acc_12x12 = self.acc2(logits_12x12, y)
        loss_16x16 = self.loss_fn(logits_16x16, y)
        acc_16x16 = self.acc3(logits_16x16, y)
        loss = loss_4x4 + loss_8x8 + loss_12x12 + loss_16x16
        out_dict = {'loss': loss,
                    'loss_4x4': loss_4x4,
                    'loss_8x8': loss_8x8,
                    'loss_12x12': loss_12x12,
                    'loss_16x16': loss_16x16,
                    'acc_4x4': acc_4x4,
                    'acc_8x8': acc_8x8,
                    'acc_12x12': acc_12x12,
                    'acc_16x16': acc_16x16
                    }
        return out_dict


    def test_step(self, batch, batch_idx):
        loss, acc = self.share_step(batch, batch_idx)
        self.log_dict({'test/loss': loss, 'test/acc': acc}, sync_dist=True, on_epoch=True)
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
        self.lr = 1e-4
        self.wd = 5e-4
        self.max_epochs = self.trainer.max_epochs

        # params_to_optimize = [self.pos_embed_width, self.pos_embed_height, list(self.net.parameters())]
        params_to_optimize = self.parameters()

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.lr,
            weight_decay=self.wd
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=2,
            max_epochs=self.max_epochs,
            warmup_start_lr=0.01 * self.lr,
            eta_min=0.01 * self.lr,
        )
        return [optimizer], [scheduler]

    def modified(self, ):
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
        self.patch_embed_12x12 = self.get_new_patch_embed(new_image_size=168, new_patch_size=12)
        self.patch_embed_16x16 = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        self.patch_embed_16x16_origin = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
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
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight,
                new_patch_size=(new_patch_size, new_patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(
                self.net.patch_embed.proj.bias.clone().detach(),
                requires_grad=True
            )

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

    wandb_logger = WandbLogger(name='MSPE-vit-jt', project='MSPE-rebuttal',
                               entity='pigpeppa', offline=False)
    checkpoint_callback = ModelCheckpoint(dirpath='ckpt/MSPE-rebuttal/MSPE-vit-jt', save_last=True)

    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])
    for image_size, patch_size in [(224, 16)]:
        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        model = ClassificationEvaluator(**args["model"])
        data_config = timm.data.resolve_model_data_config(model.net)
        transform = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                                shuffle=False, pin_memory=True)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        train_dataset = ImageFolder(root=os.path.join(args.root, 'train'), transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.works,
                                  shuffle=True, pin_memory=True)
        # trainer.test(model, dataloaders=val_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
