import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .args_generator import args

# Dataset configuration
# dataset_dirname = args.data
#
# dset_dir_candidate = [
#     args.data,
#     os.path.join('C://', 'torch_data', 'imagenet'),
#     os.path.join('E://', 'torch_data', 'imagenet'),
#     "/home/shared/ImageNet/Imagenet_2012/Imagenet_Data_tar/",
# ]
#
# for path in dset_dir_candidate:
#     if not os.path.isdir(dataset_dirname):
#         dataset_dirname = path

train_dataset_dirname = os.path.join(args.data, 'train')
val_dataset_dirname = os.path.join(args.data, 'val')

train_dset_dir_candidate = [
    os.path.join(args.data, 'train'),
    os.path.join('C://', 'torch_data', 'imagenet', 'train'),
    os.path.join('E://', 'torch_data', 'imagenet', 'train'),
    os.path.join('/home', 'shared', 'Imagenet_data', 'train', 'train'),
    os.path.join('/home', 'shared', 'Imagenet_data', 'train'),
    "/media/su8939/datasets/imagenet/train",
    os.path.join('/data', 'datasets', 'ILSVRC2012', 'train'),
]

val_dset_dir_candidate = [
    os.path.join(args.data, 'val'),
    os.path.join('C://', 'torch_data', 'imagenet', 'val'),
    os.path.join('E://', 'torch_data', 'imagenet', 'val'),
    os.path.join('/home', 'shared', 'Imagenet_data', 'val', 'val'),
    os.path.join('/home', 'shared', 'Imagenet_data', 'val'),
    "/media/su8939/datasets/imagenet/val",
    os.path.join('/data', 'datasets', 'ILSVRC2012', 'val'),
]

for path in train_dset_dir_candidate:
    if not os.path.isdir(train_dataset_dirname):
        train_dataset_dirname = path

for path in val_dset_dir_candidate:
    if not os.path.isdir(val_dataset_dirname):
        val_dataset_dirname = path

train_dataset = datasets.ImageFolder(
        train_dataset_dirname,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

val_dataset = datasets.ImageFolder(
        val_dataset_dirname,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
else:
    train_sampler = None
    val_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler)