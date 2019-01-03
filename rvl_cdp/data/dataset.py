import torch
import pandas as pd
import os
import numpy as np

from collections import OrderedDict
import rvl_cdp.data.util as data_utils

from PIL import Image
from skimage import io, transform
from skimage.transform import rescale, resize, downscale_local_mean

from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip


def one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).long()

    return y[labels]


def read_textfile(image_paths, path):
    examples = []

    with open(path, "r") as file:
        for line in file.readlines():
            image_path, label = line.split()
            image_path = os.path.join(image_paths, image_path)

            examples.append({"path": image_path, "label": int(label)})

    return pd.DataFrame(examples, columns=["path", "label"])


class HorizontalFlip:
    def __init__(self, p=0.5):
        super(HorizontalFlip, self).__init__()

        self.flip = RandomHorizontalFlip(p)

    def __call__(self, image, *args, **kwargs):
        return self.flip(image)


class VerticalFlip:
    def __init__(self, p=0.5):
        super(VerticalFlip, self).__init__()

        self.flip = RandomVerticalFlip(p)

    def __call__(self, sample, *args, **kwargs):
        image = sample["image"]

        return self.flip(image)


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomImageCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        image = np.array(image)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]
        image = Image.fromarray(image)
        return image


class Resize:
    def __init__(self, size=(256, 256)):
        super(Resize, self).__init__()
        self.size = size

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample['label']
        image = resize(image, self.size)

        return {"image": image, "label": label}


class Normalization:

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample['label']

        image = (image - image.mean()) / image.std()

        return {"image": image, "label": label}


class ToTensor:
    def __call__(self, sample, unsqueeze=False, *args, **kwargs):
        image, label = sample["image"], sample['label']
        image = torch.from_numpy(image).float()

        if unsqueeze:
            image = image.unsqueeze(0)

        return {"image": image, "label": label}

class PermuteTensor:
    def __init__(self, reordering):
        self.reordering = reordering

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample['label']

        image = image.permute(self.reordering)

        return {"image": image, "label": label}

class NPTranspose:
    def __init__(self, reordering):
        self.reordering = reordering

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample['label']

        image = np.transpose(image, self.reordering)

        return {"image": image, "label": label}


class BaseDataset(Dataset):
    def __init__(self, nb_classes, images_path=None, labels_path=None, data=None, label_dict=None,
                 transforms=None):
        super(BaseDataset, self).__init__()

        if transforms is None:
            transforms = Compose([
                Normalization(),
                Resize(),
                ToTensor()
            ])

        self.transforms = transforms
        self.nb_classes = nb_classes
        self.images_path = images_path
        self.labels_path = labels_path

        self.label_dict = label_dict if label_dict is not None else OrderedDict()
        self.data = data if data is not None else self.read_data()

    def __getitem__(self, idx):
        image_path, label = self.data.path.iloc[idx], self.data.label.iloc[idx]

        image = io.imread(image_path)

        if self.transforms:
            sample = self.transforms({"image": image, "label": label})
            image, label = sample["image"], sample['label']

        return {"image": image, "label": one_hot(label, self.nb_classes)}

    def __len__(self):
        return len(self.data)

    def read_data(self):
        pass


class RVLCDIPDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(RVLCDIPDataset, self).__init__(*args, nb_classes=16, **kwargs)

    def read_data(self):
        return read_textfile(self.images_path, self.labels_path)


class CIFAR10(BaseDataset):
    def __init__(self, *args, **kwargs):
        transforms = Compose([

            Normalization(),
            Resize(),
            ToTensor(),
            PermuteTensor((2, 0, 1))
        ])

        super(CIFAR10, self).__init__(*args, nb_classes=10, transforms=transforms, **kwargs)

    def read_data(self):
        label_dict, data = data_utils.read_image_folders(self.images_path, "*.png")

        self.label_dict = label_dict

        return data
