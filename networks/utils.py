import numpy as np
from tensorflow.image import ssim as tf_ssim
from tensorflow import reduce_mean

def psnr(gt, img, img_max):
    """

    :param gt:
    :param img:
    :param img_max:
    :return:
    """
    mse = metric_mse(gt, img)
    return 20 * np.log10(img_max) - 10 * np.log10(mse)


def metric_mse(gt, img):
    """

    :param gt:
    :param img:
    :return:
    """
    assert gt.shape == img.shape, 'GT and predicted images have different shapes.'
    mean_squared_error = (np.square(gt - img)).mean(axis=None)
    return mean_squared_error


def metric_mae(gt, img):
    """

    :param gt:
    :param img:
    :return:
    """
    assert gt.shape == img.shape, 'GT and predicted images have different shapes.'
    mean_absolute_error = (np.abs(gt - img)).mean(axis=None)
    return mean_absolute_error


def metric_ssim(gt, img, dynamic_range):
    """

    :param gt:
    :param img:
    :param dynamic_range:
    :return:
    """
    # assert gt.shape == img.shape, 'Ground truth and predicted images have different shapes.'
    mean_gt = np.mean(gt)
    mean_img = np.mean(img)
    variance_gt = np.square(np.std(gt))
    variance_img = np.square(np.std(img))
    covariance_xy = np.cov(gt.flatten(), img.flatten())[0,1]
    k1 = 0.01
    k2 = 0.03
    l = (2 ** dynamic_range) - 1
    c1 = np.square(k1 * l)
    c2 = np.square(k2 * l)

    numerator = (2 * mean_gt * mean_img + c1) * (2 * covariance_xy + c2)
    denominator = (np.square(mean_gt) + np.square(mean_img) + c1) * (variance_gt + variance_img + c2)
    ssim = numerator / denominator
    return ssim


def loss_myssim(gt, img, dynamic_range=1):
    """

    :param gt:
    :param img:
    :param dynamic_range:
    :return:
    """
    loss = (1 - metric_ssim(gt, img, dynamic_range))
    return loss.mean(axis=None)


def loss_ssim(gt, img, dynamic_range=1):
    """
    computed loss using Tensorflow's SSIM implementation, tf.image.ssim(im1, im2, dynamic_range)
    :param gt:
    :param img:
    :param dynamic_range:
    :return: loss (1-SSIM) computed by Tensorflow, tf.image.ssim(im1, im2, dynamic_range)
    """
    calc_ssim = tf_ssim(gt, img, dynamic_range)
    return 1-reduce_mean(calc_ssim)


def metric_coc(gt, img):
    """
    Coefficient of correlation used in Morales-Navarrete et al 2019 Proceedings - ICIP
    :param gt: ground truth image
    :param img: predicted image
    :return:
    """
    assert gt.shape == img.shape, 'Ground truth and predicted images have different shapes.'
    mean_gt = gt.mean()
    mean_img = img.mean()
    num = np.sum((np.subtract(gt, mean_gt)).flatten() * (np.subtract(img, mean_img)).flatten())
    denom = np.sqrt(
        np.sum((np.square(np.subtract(gt, mean_gt))).flatten()) * np.sum((np.square(np.subtract(img, mean_img))).flatten())
    )
    coc = num/denom
    return coc
