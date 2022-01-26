from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import random
import numpy as np
from TextRecognition.config import cfg
import os

CHANNEL_NUM = 3

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.random.manual_seed(0)


############# Auxiliary functions #############

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m / s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1 / s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        norm = transforms.Normalize(self.demean, self.destd)
        return norm(tensor)


def add_blanks(label):
    new_label = torch.zeros(len(label) * 2 + 1, dtype=torch.long)
    new_label[1:len(label) * 2 + 1:2] = label
    return new_label


def show_alpha_betta(alpha, betta):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    sns.heatmap(alpha[0], ax=ax)
    ax.set_title('Alpha')
    print('betta\n', betta)
    sns.heatmap(betta[0], ax=ax2)
    ax2.set_title('Betta')
    plt.show()


#############  Transformations and statistics #############

def random_hor_resize(img):
    w, h = img.size
    resize_rat = random.uniform(0.8, 1.2)
    new_w = min(int(w * resize_rat), 500)
    return img.resize((new_w, h))


def pad(img):
    """Adds aux pixel with median value to both left and right sides."""
    im_arr = np.asarray(img)
    med_val = np.median(im_arr)
    padded = np.pad(im_arr, ((0, 0), (1, 1), (0, 0)),
                    mode='constant', constant_values=med_val)

    return Image.fromarray(padded)


def gaussian_noise(image, mean=0, max_sigma=0.05):
    im_arr = np.array(image)
    height, width = im_arr.shape[:2]
    im_s = (height, width, 1)
    sigma = np.random.uniform(0, max_sigma, 1)[0]
    g = np.random.normal(mean, sigma * 255.0, im_s)
    im_arr = im_arr + g * ((im_arr + g >= 0) * (im_arr + g < 255))

    return Image.fromarray(im_arr.astype(np.uint8))


def resize(img, h_new=32):
    w, h = img.size
    w_new = min(int(w * (float(h_new) / h)), 500)
    return img.resize((w_new, h_new))


def noise(x):
    return x + torch.normal(0, 0.05, size=x.size())


class Transforms:
    @staticmethod
    def train():
        return transforms.Compose(
            [
                transforms.Lambda(resize),
                # transforms.Lambda(gaussian_noise),
                transforms.Lambda(pad),
                transforms.ToTensor(),
                # transforms.Lambda(noise),
                transforms.Normalize(cfg.images.pix_mean, cfg.images.pix_std)])

    @staticmethod
    def test():
        return transforms.Compose(
            [transforms.Lambda(resize),
             transforms.Lambda(pad),
             transforms.ToTensor(),
             transforms.Normalize(cfg.images.pix_mean, cfg.images.pix_std)])


def find_ims_stats(annot_file, nrof_files=1000):
    heights, widths = [], []
    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    with open(os.path.join(cfg.paths.dataset, annot_file), 'r') as f:
        tr_file_names = f.readlines()
        print(len(tr_file_names))

    for i, file_name in enumerate(tr_file_names[:nrof_files]):
        file_name = file_name[2:file_name.find(' ')]
        file_name = file_name.replace('/', '\\')
        im_path = os.path.join(cfg.paths.dataset, file_name)
        img = np.array(pil_loader(im_path))

        img = img / 255.0
        pixel_num += (img.size / CHANNEL_NUM)
        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

        heights.append(img.shape[0])
        widths.append(img.shape[1])

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(heights, bins='auto')
    plt.title('Heights distribution')
    plt.subplot(1, 2, 2)
    plt.hist(widths, bins='auto')
    plt.title('Widths distribution')
    plt.show()
