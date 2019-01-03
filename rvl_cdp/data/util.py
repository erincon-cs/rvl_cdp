import os
import pandas as pd

from fnmatch import fnmatch
from collections import OrderedDict


def get_files(root, pattern):
    walked_files = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                p = os.path.join(path, name)
                p = os.path.abspath(p)

                walked_files.append(p)

    return walked_files


def check_exist_path(path):
    if not os.path.exists(path):
        raise ValueError("Path {} does not exist".format(path))


def read_image_folders(images_path, image_ext_pattern):
    """
    Walks through a directory, reads the images, and returns a label dictionary and a pandas dataframe
    that has the paths and labels

    Assumes that the images are stored in the directory in seperate folders where the name of the directories are
    the name of the respective labels.

    :param images_path: (str) directory to read from
    :param image_ext_pattern: a string pattern e.g., "*.png"
    :return: an OrderedDict and a pandas DataFrame
    """

    label_dict = OrderedDict()
    image_files = get_files(images_path, pattern=image_ext_pattern)
    mapped_labels = []

    for image_file in image_files:
        dirs = image_file.split("/")
        label = dirs[len(dirs) - 2]
        label = label.lower().strip()

        if label not in label_dict:
            label_dict[label] = len(label_dict)
        mapped_label = label_dict[label]

        mapped_labels.append(mapped_label)

    return label_dict, pd.DataFrame({"path": image_files, "label": mapped_labels})
