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
        ).to(self.device)
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def forward_patch_embed(self, x):
        x = self.net.patch_embed(x)
        return x

    def forward_class_token(self, x):
        x = self.net.forward_features(x)
        if self.net.attn_pool is not None:
            x = self.net.attn_pool(x)
        elif self.net.global_pool == 'avg':
            x = x[:, self.net.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        return x
        # x = self.fc_norm(x)
        # x = self.head_drop(x)
        # return x if pre_logits else self.head(x)

    def forward_patch_stats(self, x):
        """Extracts the patch embeddings and computes their mean and variance."""
        patch_embeds = self.net.patch_embed(x)  # Assuming patch_embed is accessible like this
        return patch_embeds.mean(dim=[0, 2, 3]), patch_embeds.var(dim=[0, 2, 3])

    def forward_class_token_stats(self, x):
        """Extracts the class token and computes its mean and variance."""
        class_token = self.forward_class_token(x)  # Reuse your forward_class_token method
        return class_token.mean(dim=0), class_token.var(dim=0)



    def test_step(self, batch, _):
        x, y = batch
        x = F.interpolate(x, size=self.image_size, mode='bilinear')

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        acc = self.acc(pred, y)

        # Log
        # self.log_dict({'test_loss': loss, 'test_acc': acc}, sync_dist=True, on_epoch=True)

        # Compute stats for patch embeddings and class tokens
        patch_mean, patch_var = self.forward_patch_stats(x)
        class_token_mean, class_token_var = self.forward_class_token_stats(x)

        # Log
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc,
            'patch_mean': patch_mean,
            'patch_variance': patch_var,
            'class_token_mean': class_token_mean,
            'class_token_variance': class_token_var
        }, sync_dist=True, on_epoch=True)

        return {
            'loss': loss,
            'accuracy': acc,
            'patch_mean': patch_mean,
            'patch_variance': patch_var,
            'class_token_mean': class_token_mean,
            'class_token_variance': class_token_var
        }

        # return loss

    def test_epoch_end(self, outputs):

        # Example of how to aggregate means and variances across batches
        avg_patch_mean = torch.stack([x['patch_mean'] for x in outputs]).mean(0)
        avg_patch_variance = torch.stack([x['patch_variance'] for x in outputs]).mean(0)
        avg_class_token_mean = torch.stack([x['class_token_mean'] for x in outputs]).mean(0)
        avg_class_token_variance = torch.stack([x['class_token_variance'] for x in outputs]).mean(0)

        # Log or print final aggregated values
        print("Average Patch Mean:", avg_patch_mean)
        print("Average Patch Variance:", avg_patch_variance)
        print("Average Class Token Mean:", avg_class_token_mean)
        print("Average Class Token Variance:", avg_class_token_variance)

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


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--works", type=int, default=4)
    parser.add_argument("--root", type=str, default='./data')
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts

    # wandb_logger = WandbLogger(name='test', project='flexivit', entity='pigpeppa', offline=False)
    # trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer = pl.Trainer.from_argparse_args(args)
    # for image_size, patch_size in [(32, 4), (48, 4), (64, 4), (80, 8), (96, 8), (112, 8), (128, 8), (144, 16),
    #                                (160, 16), (176, 16), (192, 16), (208, 16), (224, 16)]:
    # for image_size, patch_size in [(56, 4),(112, 8)]:
    for image_size, patch_size in [(56, 4), (224, 16)]:
        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        model = ClassificationEvaluator(**args["model"])
        data_config = timm.data.resolve_model_data_config(model.net)
        transform = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                                shuffle=False, pin_memory=True)
        trainer.test(model, dataloaders=val_loader)
