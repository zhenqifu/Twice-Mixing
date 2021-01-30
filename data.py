from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize
from dataset import DatasetFromFolderEval, DatasetFromFolder


def transform():
    return Compose([
        Resize((128, 128)),
        ToTensor(),
    ])


def transform2():
    return Compose([
        ToTensor(),
    ])


def get_training_set(data_dir, im0, im1, im2):
    im0_dir = join(data_dir, im0)
    im1_dir = join(data_dir, im1)
    im2_dir = join(data_dir, im2)
    return DatasetFromFolder(im0_dir, im1_dir,  im2_dir, transform=transform())


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir,  transform=transform2())

