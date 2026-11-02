"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape


def LoadHippocampusData(root_dir, y_shape, z_shape):
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")

    images = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and f[0] != "."]

    out = []
    for f in images:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        image = image.astype(np.float32)
        img_min, img_max = image.min(), image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)

        # TASK: med_reshape function is not complete. Go and fix it!
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: CrossEntropyLoss requires integer class indices (0, 1, 2).

        out.append({"image": image, "seg": label, "filename": f})

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
