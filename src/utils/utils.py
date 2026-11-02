"""
Various utility methods in this module
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from PIL import Image

mpl.use("agg")


def mpl_image_grid(images):
    n = min(images.shape[0], 16)
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2 * rows, 2 * cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            vol = images[i].detach().numpy()
            img = [[[(1 - vol[0, x, y]) * vol[1, x, y],
                      (1 - vol[0, x, y]) * vol[2, x, y], 0]
                     for y in range(vol.shape[2])]
                    for x in range(vol.shape[1])]
            plt.imshow(img)
        else:
            plt.imshow((images[i, 0] * 255).int(), cmap="gray")
    return figure


def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):
    writer.add_scalar("Loss", loss, counter)
    writer.add_figure("Image Data", mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask", mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map", mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction", mpl_image_grid(
        torch.argmax(prediction, dim=1, keepdim=True).float().cpu() / 2.0), global_step=counter)


def save_numpy_as_image(arr, path):
    im = Image.fromarray(arr)
    im.save(path)


def med_reshape(image, new_shape):
    """
    Reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner
    """
    reshaped_image = np.zeros(new_shape, dtype=image.dtype)
    # TASK: write your original image into the reshaped image
    reshaped_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image
    return reshaped_image
