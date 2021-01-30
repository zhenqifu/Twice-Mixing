import torch.utils.data as data
import os
from os import listdir
from os.path import join
from PIL import Image
import random
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, im0_dir, im1_dir, im2_dir, transform=None):
        super(DatasetFromFolder, self).__init__()

        im0_filenames = [join(im0_dir, x) for x in listdir(im0_dir) if is_image_file(x)]
        im0_filenames.sort()
        self.im0_filenames = im0_filenames

        im1_filenames = [join(im1_dir, x) for x in listdir(im1_dir) if is_image_file(x)]
        im1_filenames.sort()

        im2_filenames = [join(im2_dir, x) for x in listdir(im2_dir) if is_image_file(x)]
        im2_filenames.sort()
        self.im1_filenames = im1_filenames + im2_filenames

        self.transform = transform

    def __getitem__(self, index):
        k1 = 0
        k2 = 0
        while abs(k1/10 - k2/10) < 0.1:
            k1 = random.randint(0, 10)
            k2 = random.randint(0, 10)

        k1 = k1/10
        k2 = k2/10
        base_path = '../dataset/train/original/'
        file_type = '.jpg'

        a = self.im1_filenames[index]
        b = a.split('_', 1)
        c = b[0].split('/')
        name = base_path + c[-1] + file_type

        I0 = load_img(name)
        I1 = load_img(self.im1_filenames[index])

        if c[3] == 'high-quality':
            I2 = I1
            I1 = I0
            I0 = I2

        im0 = k1 * np.array(I0) + (1 - k1) * np.array(I1)
        im1 = k2 * np.array(I0) + (1 - k2) * np.array(I1)

        im0 = Image.fromarray(np.uint8(im0))
        im1 = Image.fromarray(np.uint8(im1))

        if k1 > k2:
            score = 1
        else:
            score = -1

        _, file = os.path.split(self.im1_filenames[index])

        if self.transform:
            im0 = self.transform(im0)
            im1 = self.transform(im1)

        return im0, im1, score

    def __len__(self):
        return len(self.im1_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        filenames.sort()
        self.filenames = filenames
        self.transform = transform

    def __getitem__(self, index):

        im = load_img(self.filenames[index])
        _, file = os.path.split(self.filenames[index])

        if self.transform:
            im = self.transform(im)
        return im, file

    def __len__(self):
        return len(self.filenames)
