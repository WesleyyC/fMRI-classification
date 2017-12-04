import numpy as np
from scipy.ndimage import gaussian_filter


def normalized_data(images, sigma=1):
    normed = np.zeros(images.shape)
    for idx, image in enumerate(images):
        normed[idx] = gaussian_filter(image, sigma=sigma)
        normed[idx] = np.multiply(normed[idx], (normed[idx] > np.percentile(normed[idx].flatten(), 90)))
    return normed
