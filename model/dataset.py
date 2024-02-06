import os
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
import os.path
import random
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageDraw


class Image_Editing_Dataset(Dataset):
    def __init__(self, cfg, dataset_root, split='test', dataset_name=''):
        self.split = split
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.img_format = '.png'

        self.dir_img = os.path.join(dataset_root, 'test_processed', 'images')
        self.dir_lab = os.path.join(dataset_root, 'test_processed', 'labels')
        self.dir_ins = os.path.join(dataset_root, 'test_processed', 'inst_map')
        name_list = os.listdir(self.dir_img)
        self.name_list = [n[:-4] for n in name_list if n.endswith(self.img_format)]
        self.name_list.sort()
        self.predefined_mask_path = os.path.join(dataset_root, f'test_processed', 'predefined_masks')

    def __getitem__(self, index):
        name = self.name_list[index]
        # input data
        img = cv2.imread(os.path.join(self.dir_img, name + '.png'))
        lab = cv2.imread(os.path.join(self.dir_lab, name + '.png'), 0)
        inst_map = Image.open(os.path.join(self.dir_ins, name + '.png'))
        inst_map = np.array(inst_map, dtype=np.int32)

        assert len(inst_map.shape) == 2
        
        img = get_transform(img)
        lab = get_transform(lab, normalize=False)
        lab = lab * 255.0

        mask = cv2.imread(os.path.join(self.predefined_mask_path, 'type_0', name + '.png'), 0) / 255
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)
        
        mask = torch.from_numpy(mask)
        masked_img = img * (1. - mask)

        inst_map = inst_map.reshape((1,) + inst_map.shape).astype(np.float32)
        inst_map = torch.from_numpy(inst_map)
        
        return {'img': img, 'masked_img': masked_img, 'lab': lab, 'mask': mask, 'inst_map': inst_map, 'name': name}
        # 'mask_seam': mask_seam,

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.name_list)


def get_transform(img, normalize=True):
    transform_list = []

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)(img)
