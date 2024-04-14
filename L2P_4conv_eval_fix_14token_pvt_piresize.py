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
from timm.models._manipulate import checkpoint_seq

import torch.nn as nn
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
        self.acc_0 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_1 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_2 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)
        self.acc_3 = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

        # modified
        self.modified()

    def test_step(self, batch, _):
        x, y = batch
        x = F.interpolate(x, size=self.image_size, mode='bilinear')

        # Pass through network
        pred0, pred1, pred2, pred3 = self.ms_forward(x)

        # Get accuracy
        acc_0 = self.acc_0(pred0, y)
        acc_1 = self.acc_1(pred1, y)
        acc_2 = self.acc_2(pred2, y)
        acc_3 = self.acc_3(pred3, y)

        # Log
        out_dict = {
            'res': self.image_size,
            'test_acc_0': acc_0,
            'test_acc_1': acc_1,
            'test_acc_2': acc_2,
            'test_acc_3': acc_3
        }
        self.log_dict(out_dict, sync_dist=True, on_epoch=True)

        return out_dict

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

    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.net.forward_head(x)
        return x

    def forward_after_patch_embed(self, x):
        x = self.net.patch_embed.norm(x)
        x = self.net.stages(x)
        x = self.net.forward_head(x)
        return x

    def ms_forward(self, x):
        # x_3x3 = F.interpolate(x, size=56, mode='bilinear')
        x_3x3 = self.patch_embed_3x3_s1(x, patch_size=3, stride=1)

        # x_5x5 = F.interpolate(x, size=112, mode='bilinear')
        x_5x5 = self.patch_embed_5x5_s2(x, patch_size=5, stride=2)

        # x_7x7 = F.interpolate(x, size=168, mode='bilinear')
        x_7x7 = self.patch_embed_7x7_s3(x, patch_size=7, stride=3)

        if self.image_size > 28:
            x_7x7_s4 = F.interpolate(x, size=224, mode='bilinear')
            x_7x7_s4 = self.patch_embed_7x7_s4(x_7x7_s4, patch_size=7, stride=4)
            return self.forward_after_patch_embed(x_3x3), \
                self.forward_after_patch_embed(x_5x5), \
                self.forward_after_patch_embed(x_7x7), \
                self.forward_after_patch_embed(x_7x7_s4)

        else:
            return self.forward_after_patch_embed(x_3x3), \
                self.forward_after_patch_embed(x_5x5), \
                self.forward_after_patch_embed(x_7x7), \
                torch.zeros(x.size(0), self.num_classes, device=x.device)
        # x_7x7_s4 = self.patch_embed_7x7_s4(x, patch_size=7, stride=4)

    def modified(self):
        self.in_chans = 3
        self.embed_dim = 64
        # import pdb;pdb.set_trace()
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
    parser.add_argument("--ckpt_path", type=str, default='./ckpt')
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts
    trainer = pl.Trainer.from_argparse_args(args)

    # results_path = f"./L2P_exp/{args.ckpt_path.split('/')[-2]}_fix_14token.csv"
    results_path = f"./L2P_exp/PVT_fix_14token_piresize.csv"
    print(f'result save in {results_path} ...')
    if os.path.exists(results_path):
        print(f'exist {results_path}, removing ...')
        os.remove(results_path)

    for image_size, patch_size in [(28, 2), (42, 3), (56, 4), (70, 5), (84, 6), (98, 7), (112, 8), (126, 9), (140, 10),
                                   (154, 11), (168, 12), (182, 13), (196, 14), (210, 15), (224, 16), (238, 17),
                                   (252, 18)]:
        # for image_size, patch_size in [(42, 3), (56, 4), (70, 5), (84, 6), (98, 7), (112, 8), (126, 9),(140, 10),
        #                                (154, 11), (168, 12), (182, 13), (196, 14), (210, 15), (224, 16), (238, 17),
        #                                (252, 18)]:
        # for image_size, patch_size in [(28, 2)]:

        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        args["model"].results_path = results_path
        model = ClassificationEvaluator(**args["model"])
        data_config = timm.data.resolve_model_data_config(model.net)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                                shuffle=False, pin_memory=True)
        trainer.test(model, dataloaders=val_loader)
