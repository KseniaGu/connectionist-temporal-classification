import os
import pickle
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import random
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import pil_loader, Transforms, Denormalize

torch.random.manual_seed(0)


class mjsynthDataset(Dataset):
    def __init__(self, image_paths, labels, transforms, config):
        self.config = config
        self.image_paths = image_paths
        self.alphabet = ('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        self.alphabet = dict(zip(self.alphabet, range(len(self.alphabet))))
        self._make_encoded_labels(labels)
        self.transforms = transforms

    def _make_encoded_labels(self, labels):
        self.labels = []
        for label in labels:
            self.labels.append([self.alphabet[c] + 1 for c in label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns transformed image, coded label and coded label length."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.transforms(pil_loader(self.image_paths[idx])),
                torch.tensor(self.labels[idx]), torch.tensor(len(self.labels[idx])))


def my_collate_fn(batch):
    """Pads image to max batch width with arbitrary pixels by both side."""
    images = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    lens = [sample[2] for sample in batch]

    images_max_width = max(im.shape[-1] for im in images)

    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    # random_padding_lens = (random.randint(0,(images_max_width - im.shape[-1])) for im in images)
    images = [transforms.functional.pad(im, (0, 0,
                                             images_max_width - im.shape[-1], 0),
                                        padding_mode='edge') for im in images]

    return torch.stack(images), labels, torch.stack(lens)


class Data_loader:
    def __init__(self, config):
        self.config = config
        self._load_data()

    def check_corrupted_images(self, images_dir, to_save=True):
        if os.path.exists(self.config.paths.corrupted_images_paths):
            with open(self.config.paths.corrupted_images_paths, 'rb') as f:
                not_valid_files = pickle.load(f)
        else:
            not_valid_files = []

        for x in tqdm(os.listdir(images_dir)):
            if x in not_valid_files:
                continue
            try:
                with open(os.path.join(images_dir, x), 'rb') as f:
                    Image.open(f)
            except Exception as e:
                print(e)
                not_valid_files.append(x)

        if to_save:
            with open(self.config.paths.corrupted_images_paths, 'wb') as f:
                pickle.dump(not_valid_files, f)

        return not_valid_files

    def make_paths_and_labels(self, phase='train'):
        """Goes through annotation_<phase> files from dataset directory and gathers images paths."""
        image_paths, labels = [], []
        images_dir = getattr(self.config.paths, phase + '_split_dir')
        not_valid_files = self.check_corrupted_images(images_dir)

        for x in os.listdir(images_dir)[:self.config.train.train_size]:
            if not x in not_valid_files:
                image_paths.append(os.path.join(images_dir, x))
                labels.append(x.split('_')[1])

        return image_paths, labels

    def _load_data(self):
        tr_image_paths, tr_labels = self.make_paths_and_labels(phase='train')
        val_image_paths, val_labels = self.make_paths_and_labels(phase='val')
        test_image_paths, test_labels = self.make_paths_and_labels(phase='test')

        self.tr_dataset = mjsynthDataset(tr_image_paths, tr_labels, Transforms.train(self.config), self.config)
        self.val_dataset = mjsynthDataset(val_image_paths, val_labels, Transforms.test(self.config), self.config)
        self.test_dataset = mjsynthDataset(test_image_paths, test_labels, Transforms.test(self.config), self.config)

        self.tr_loader = DataLoader(self.tr_dataset, batch_size=self.config.train.batch_size,
                                    collate_fn=my_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.train.batch_size,
                                     collate_fn=my_collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.train.batch_size,
                                      collate_fn=my_collate_fn)

    def check_images(self, train=True, nrof_to_check=4):
        dataset = self.tr_dataset if train else self.val_dataset

        for i in range(4):
            random_ims_ids = random.sample(range(len(dataset)), nrof_to_check)
            plt.figure(figsize=(10, 10))
            for i in range(nrof_to_check):
                D = Denormalize(mean=self.config.images.pix_mean, std=self.config.images.pix_std)
                denorm = D(dataset[random_ims_ids[i]][0]).numpy()
                ax = plt.subplot(2, 2, i + 1)
                plt.imshow(denorm.transpose(1, 2, 0))
            plt.show()

    def check_iterations(self, train=False):
        random_batches_nums = random.choices(range(len(self.tr_loader)), k=4)
        j = 0
        t = time()

        for i, batch in enumerate(self.tr_loader):
            if i in random_batches_nums:
                D = Denormalize(mean=self.config.images.pix_mean, std=self.config.images.pix_std)
                images, labels, lens = batch
                denorm = D(images[0]).numpy()
                plt.subplot(2, 2, j + 1)
                plt.imshow(denorm.transpose(1, 2, 0))
                j += 1

        print(time() - t)
        plt.show()
