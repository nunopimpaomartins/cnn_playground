import numpy as np


def psnr(gt, img, img_max):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(img_max) - 10 * np.log10(mse)
