import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
import timm
from tqdm import tqdm


def main_worker(gpu, ngpus_per_node, args):
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=ngpus_per_node, rank=gpu)

    # 创建模型并移动到对应的GPU
    model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # 加载ImageNet验证集
    val_dataset = ImageFolder(root=args.imagenet_val_dir, transform=args.transform)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, sampler=val_sampler)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating GPU {gpu}"):
            images, labels = images.cuda(gpu, non_blocking=True), labels.cuda(gpu, non_blocking=True)
            outputs = model(images)
            _, predicted_top5 = outputs.topk(5, 1, True, True)
            predicted_top1 = predicted_top5[:, :1]

            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels.view(-1, 1)).sum().item()
            correct_top5 += (predicted_top5 == labels.view(-1, 1)).any(dim=1).sum().item()

    # 输出准确率等信息
    if gpu == 0:
        print(f'Top-1 Accuracy: {100 * correct_top1 / total:.2f}%')
        print(f'Top-5 Accuracy: {100 * correct_top5 / total:.2f}%')


def main():
    ngpus_per_node = torch.cuda.device_count()

    args = type('', (), {})()  # 创建一个空对象用于存储参数
    args.batch_size = 256
    args.workers = 4
    args.imagenet_val_dir = '/mnt/mmtech01/dataset/lzy/ILSVRC2012/val'
    data_config = timm.data.resolve_model_data_config(model)
    args.transform = timm.data.create_transform(**data_config, is_training=False)

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == '__main__':
    main()
