import numpy as np


def normalized_data(images):
    normed = np.zeros(images.shape)
    for idx, image in enumerate(images):
        mean = np.mean(image)
        std = np.std(image)
        normed[idx] = (image - mean)/std
    return normed
