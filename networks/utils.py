import numpy as np


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
    assert gt.shape == img.shape, 'Ground truth and predicted images have different shapes.'
    mean_gt = gt.mean()
    mean_img = img.mean()
    variance_gt = np.square(np.std(gt))
    variance_img = np.square(np.std(img))
    covariance_xy = np.cov(gt.flatten(), img.flatten())
    k1 = 0.01
    k2 = 0.03
    # l = (np.power(2, dynamic_range - 1))
    l = (2 ^ dynamic_range - 1)
    c1 = np.square(k1 * l)
    c2 = np.square(k2 * l)

    numerator = (2 * mean_gt * mean_img + c1) * (2 * covariance_xy + c2)
    denominator = (np.square(mean_gt) + np.square(mean_img) + c1) * (variance_gt + variance_img + c2)
    ssim = numerator / denominator
    return ssim


def metric_coc(gt, img):
    """
    Coefficient of correlation used in Morales-Navarrete et al 2019 Proceedings - ICIP
    :param gt: ground truth image
    :param img: predicted image
    :return:
    """
    mean_gt = gt.mean()
    mean_img = img.mean()
    sum_numerator = 0
    # for i in range()