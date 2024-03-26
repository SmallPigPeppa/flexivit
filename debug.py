from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import timm
from tqdm import tqdm

# ImageNet验证集路径
imagenet_val_dir = '/mnt/mmtech01/dataset/lzy/ILSVRC2012/val'  # 需要替换为实际路径

# 创建模型
model = timm.create_model('vit_base_patch16_224', pretrained=True)
# vit_base_patch16_224.augreg_in21k_ft_in1k
model.eval()
model.cuda()  # 使用GPU

# 获取模型特定的transforms
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# 加载ImageNet验证集
val_dataset = ImageFolder(root=imagenet_val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# 准确率计算
correct_top1 = 0
correct_top5 = 0
total = 0

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
