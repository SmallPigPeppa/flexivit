import os
from typing import Callable, Optional, Sequence, Union
import pandas as pd
import torch
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from torch.nn import CosineSimilarity, CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.models._manipulate import checkpoint_seq
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
import numpy as np


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str,
            num_classes: int = 1000,
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
        self.cosine_similarity = CosineSimilarity(dim=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

        # modified
        self.modified()

        self.sim_patches_list = []
        self.sim_classes_list = []

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
        x = self.forward_patch_embed(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_56(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_patch_embed_56(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_224(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_patch_embed_224(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def modified(self):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            # flatten deferred until after pos embed
            # pass
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
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight, new_patch_size=(new_patch_size, new_patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(self.net.patch_embed.proj.bias.clone().detach(),
                                                     requires_grad=True)

        return new_patch_embed

    def forward_patch_embed(self, x):
        if self.image_size == 56:
            x = self.patch_embed_4x4(x, patch_size=self.patch_size)
        elif self.image_size == 224:
            x = self.patch_embed_16x16_origin(x, patch_size=self.patch_size)
        return x

    def forward_patch_embed_56(self, x):
        x = self.patch_embed_4x4(x, patch_size=4)
        return x

    def forward_patch_embed_224(self, x):
        x = self.patch_embed_16x16_origin(x, patch_size=16)
        return x

    def forward_class_token_56(self, x):
        x = self.forward_patch_embed_56(x)
        x = self.forward_features(x)
        if self.net.attn_pool is not None:
            x = self.net.attn_pool(x)
        elif self.net.global_pool == 'avg':
            x = x[:, self.net.num_prefix_tokens:].mean(dim=1)
        elif self.net.global_pool:
            x = x[:, 0]  # class token
        return x

    def forward_class_token_224(self, x):
        x = self.forward_patch_embed_224(x)
        x = self.forward_features(x)
        if self.net.attn_pool is not None:
            x = self.net.attn_pool(x)
        elif self.net.global_pool == 'avg':
            x = x[:, self.net.num_prefix_tokens:].mean(dim=1)
        elif self.net.global_pool:
            x = x[:, 0]  # class token
        return x

    def sim(self, x):
        x_224 = F.interpolate(x, size=224, mode='bilinear')
        x_56 = F.interpolate(x, size=56, mode='bilinear')

        patch_embed_56 = self.forward_patch_embed_56(x_56)
        patch_embed_224 = self.forward_patch_embed_224(x_224)
        flat1 = patch_embed_56.reshape(1, -1)
        flat2 = patch_embed_224.reshape(1, -1)

        # 计算余弦相似度
        sim_patches = self.cosine_similarity(flat1, flat2)

        class_token_56 = self.forward_class_token_56(x_56)
        class_token_224 = self.forward_class_token_224(x_224)
        flat1 = class_token_56.reshape(1, -1)
        flat2 = class_token_224.reshape(1, -1)
        sim_classes = self.cosine_similarity(flat1,flat2)

        return sim_patches.item(), sim_classes.item()

    def test_step(self, batch, _):
        x, y = batch
        x_224 = F.interpolate(x, size=224, mode='bilinear')
        x_56 = F.interpolate(x, size=56, mode='bilinear')

        # Pass through network
        pred_56 = self.forward_56(x_56)
        acc_56 = self.acc(pred_56, y)
        pred_224 = self.forward_224(x_224)
        acc_224 = self.acc(pred_224, y)
        self.log_dict({'acc_56': acc_56, 'acc_224': acc_224}, sync_dist=True, on_epoch=True)

        sim_patches, sim_classes = self.sim(x)
        self.sim_patches_list.append(sim_patches)
        self.sim_classes_list.append(sim_classes)

        # Append results to DataFrame if needed
        results = {'Sim Patches': sim_patches, 'Sim Classes': sim_classes}
        results_df = pd.DataFrame([results])
        results_df.to_csv(self.results_path, mode='a', header=not os.path.exists(self.results_path))
        return results

    def test_epoch_end(self, outputs):
        # Calculate mean and variance for sim_patches and sim_classes
        patches_array = np.array(self.sim_patches_list)
        classes_array = np.array(self.sim_classes_list)

        patches_mean = np.mean(patches_array)
        patches_var = np.var(patches_array)
        classes_mean = np.mean(classes_array)
        classes_var = np.var(classes_array)

        print(f"Mean and Variance of Sim Patches: Mean = {patches_mean}, Variance = {patches_var}")
        print(f"Mean and Variance of Sim Classes: Mean = {classes_mean}, Variance = {classes_var}")




if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--works", type=int, default=4)
    parser.add_argument("--root", type=str, default='./data')
    parser.add_argument("--ckpt_path", type=str, default='./ckpt')
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts
    trainer = pl.Trainer.from_argparse_args(args)

    results_path = "exp_vis/debug.csv"
    print(f'result save in {results_path} ...')
    if os.path.exists(results_path):
        print(f'exist {results_path}, removing ...')
        os.remove(results_path)

    args["model"].results_path = results_path
    model = ClassificationEvaluator.load_from_checkpoint(checkpoint_path=args.ckpt_path, strict=True,
                                                         **args["model"])
    data_config = timm.data.resolve_model_data_config(model.net)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=val_transform)

    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,shuffle=True, pin_memory=True)
    # trainer.test(model, dataloaders=val_loader)
    # Set random seed for reproducibility
    np.random.seed(42)
    subset_indices = np.random.choice(len(val_dataset), 100, replace=False)
    subset_dataset = torch.utils.data.Subset(val_dataset, subset_indices)

    val_loader = DataLoader(subset_dataset, batch_size=args.batch_size, num_workers=args.works, shuffle=False,
                            pin_memory=True)

    trainer.test(model, dataloaders=val_loader)
