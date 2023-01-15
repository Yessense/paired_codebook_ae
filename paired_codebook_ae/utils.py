import os

import torch
from functools import reduce
import operator


def find_best_model(ckpt_dir):
    if not os.path.exists(ckpt_dir) or not os.path.isdir(ckpt_dir):
        raise FileExistsError(f"This directory is not exists: {ckpt_dir}")

    ckpt_files = os.listdir(ckpt_dir)
    for filename in ckpt_files:
        if filename.startswith("best"):
            return os.path.join(ckpt_dir, filename)
    else:
        raise FileExistsError(f"Best model is not exists in dir: {ckpt_dir}")


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs > 0.5
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0,
                              10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()


def product(arr):
    return reduce(operator.mul, arr)


if __name__ == '__main__':
    print(product([3, 4, 5]))
