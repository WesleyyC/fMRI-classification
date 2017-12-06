import numpy as np
from scipy.ndimage import gaussian_filter


def normalized_data(images, sigma=1):
    normed = np.zeros(images.shape)
    for idx, image in enumerate(images):
        normed[idx] = image
        # normed[idx] = gaussian_filter(normed[idx], sigma=sigma)
        # normed[idx] = np.multiply(normed[idx], (normed[idx] > np.percentile(normed[idx].flatten(), 80)))
    return normed
