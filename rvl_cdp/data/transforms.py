import numpy as np
import torch
from PIL import Image
from skimage import transform
from skimage.transform import resize
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip


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
    def __init__(self, unsqueeze=False):
        self.unsqueeze = unsqueeze

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample['label']
        image = torch.from_numpy(image).float()

        if self.unsqueeze:
            image = image.unsqueeze(2)

        return {"image": image, "label": label}

class UnSqueeze:
    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample['label']

        image = image.unsqueeze(1)

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