import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.utils import shuffle


def normalized_data(images, sigma=1):
    normed = np.zeros(images.shape)
    for idx, image in enumerate(images):
        normed[idx] = image
        normed[idx] = gaussian_filter(normed[idx], sigma=sigma)
        # normed[idx] = np.multiply(normed[idx], (normed[idx] > np.percentile(normed[idx].flatten(), 80)))
    return normed


def resample_data(X, Y):
    label_distr = np.sum(Y, axis=0)
    label_freq = np.percentile(label_distr, 100) / label_distr

    resample_Y = []
    resample_X = []

    for i in range(len(X)):

        if Y[i][1] == 1 or Y[i][13] == 1:
            count = 10
        elif Y[i][18]:
            count = 3
        elif Y[i][11]:
            count = 3
        elif Y[i][2]:
            count = 2
        elif Y[i][6]:
            count = 2
        elif Y[i][8]:
            count = 2
        elif Y[i][10]:
            count = 2
        else:
            count = 1

        for _ in range(count):
            resample_Y.append(Y[i])
            resample_X.append(X[i])

    resample_X = np.array(resample_X)
    resample_Y = np.array(resample_Y)

    return shuffle(resample_X, resample_Y)
