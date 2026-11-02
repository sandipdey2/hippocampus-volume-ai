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

        # Normalize image to [0, 1]
        image = image.astype(np.float32)
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = image * 0.0

        # Reshape / crop / pad
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(np.int64)

        out.append({"image": image, "seg": label, "filename": f})

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)