import os
import random
import numpy as np
from PIL import Image
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image_file(filename):
    # ------------------------------------------------------
    # https://www.runoob.com/python/python-func-any.html
    # https://www.runoob.com/python/att-string-endswith.html
    # ------------------------------------------------------
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


def image_transforms(load_size, crop_size):
    # --------------------------------------------------------------
    # https://blog.csdn.net/weixin_38533896/article/details/86028509
    # --------------------------------------------------------------
    return transforms.Compose([
        # transforms.Resize(size=load_size, interpolation=Image.BICUBIC),
        # transforms.RandomCrop(size=crop_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])


def mask_transforms(crop_size):

    return transforms.Compose([
        # transforms.Resize(size=crop_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

def gen8pic(img):
    width, height = img.size

    # 计算新的宽度和高度，确保它们是2的n次方
    new_width = 2 ** math.ceil(math.log2(width))
    new_height = 2 ** math.ceil(math.log2(height))

    # 创建一个新的黑色背景图像
    new_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # 计算原始图像在新图像中的位置，使其中心对齐
    left = (new_width - width) // 2
    top = (new_height - height) // 2

    # 将原始图像粘贴到新图像上
    new_img.paste(img, (left, top))

    return new_img


class ImageDataset(Dataset):

    def __init__(self, image_root, mask_root, load_size, crop_size):
        super(ImageDataset, self).__init__()

        self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
                            for file in files if check_image_file(file)]
        self.mask_files = [os.path.join(root, file) for root, dirs, files in os.walk(mask_root)
                           for file in files if check_image_file(file)]

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)
        self.load_size = load_size
        self.crop_size = crop_size
        self.image_files_transforms = image_transforms(load_size, crop_size)
        self.mask_files_transforms = mask_transforms(crop_size)

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        mask = Image.open(self.mask_files[index % self.number_mask])

        image = gen8pic(image)
        mask = gen8pic(mask)

        ground_truth = self.image_files_transforms(image.convert('RGB'))
        mask = self.mask_files_transforms(mask.convert('RGB'))

        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        # ---------------------------------------------------
        # white values(ones) denotes the area to be inpainted
        # dark values(zeros) is the values remained
        # ---------------------------------------------------
        mask = 1 - mask
        input_image = ground_truth * mask

        return input_image, ground_truth, mask

    def __len__(self):
        return self.number_image
    
    @staticmethod
    def collate_fn(batch):
        images, targets, masks = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        batched_masks = cat_list(masks, fill_value=0)
        return batched_imgs, batched_targets, batched_masks

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

class ImageDataset2(Dataset):

    def __init__(self, image_root, mask_root, load_size, crop_size):
        super(ImageDataset2, self).__init__()

        self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
                            for file in files if check_image_file(file)]
        self.mask_files = [os.path.join(root, file) for root, dirs, files in os.walk(mask_root)
                           for file in files if check_image_file(file)]

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)
        self.load_size = load_size
        self.crop_size = crop_size
        self.image_files_transforms = image_transforms(load_size, crop_size)
        self.mask_files_transforms = mask_transforms(crop_size)
        self.root_dir = image_root
        self.file_names = os.listdir(image_root)

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        mask = Image.open(self.mask_files[index % self.number_mask])

        image = gen8pic(image)
        mask = gen8pic(mask)

        ground_truth = self.image_files_transforms(image.convert('RGB'))
        mask = self.mask_files_transforms(mask.convert('RGB'))

        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        # ---------------------------------------------------
        # white values(ones) denotes the area to be inpainted
        # dark values(zeros) is the values remained
        # ---------------------------------------------------
        mask = 1 - mask
        input_image = ground_truth * mask

        return input_image, ground_truth, mask, self.file_names[index]

    def __len__(self):
        return self.number_image
    
    @staticmethod
    def collate_fn(batch):
        images, targets, masks = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        batched_masks = cat_list(masks, fill_value=0)
        return batched_imgs, batched_targets, batched_masks

