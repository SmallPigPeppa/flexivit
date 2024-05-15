from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import timm
from tqdm import tqdm

# ImageNet验证集路径
imagenet_val_dir = '/ppio_net0/torch_ds/imagenet/val'  # 需要替换为实际路径

# 创建模型
# model = timm.create_model('vit_base_patch16_224', pretrained=True)
model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
# vit_base_patch16_224.augreg_in21k_ft_in1k
model.eval()
model.cuda()  # 使用GPU

# 获取模型特定的transforms
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

from torchvision.transforms import InterpolationMode
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
])

# 加载ImageNet验证集
val_dataset = ImageFolder(root=imagenet_val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# 准确率计算
correct_top1 = 0
correct_top5 = 0
total = 0

import torchvision

model = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
model.eval()
model.cuda()

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted_top5 = outputs.topk(5, 1, True, True)
        predicted_top1 = predicted_top5[:, :1]

        total += labels.size(0)
        correct_top1 += (predicted_top1 == labels.view(-1, 1)).sum().item()
        correct_top5 += (predicted_top5 == labels.view(-1, 1)).any(dim=1).sum().item()

print(f'Top-1 Accuracy: {100 * correct_top1 / total:.2f}%')
print(f'Top-5 Accuracy: {100 * correct_top5 / total:.2f}%')
