from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.nnFormer.MemorynnFormer_seg import ma_nnFormer
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from networks.SwinUnetr.swin_unetr import MemorySwinUNETR
from networks.MAUNetr.ma_unetr import MA_UNETR
# from networks.TransBTS.MATransBTS_downsample8x_skipconnection import MemoryTransBTS
import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms, infer_post_transforms_labels
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
args = parser.parse_args()

args.root = r'E:\12_李国润\MA\3DUX-Net-main\3DUX-Net-main\dataset\FeTA2021'
args.dataset = 'feta'
args.mode = 'train'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

set_determinism(seed=0)

train_samples, valid_samples, out_classes = data_loader(args)

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

## Valid Pytorch Data Loader and Caching
train_transforms, val_transforms = data_transforms(args)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


for step, batch in enumerate(val_loader):
    val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
    print(val_inputs.shape)



