import os

import cv2
import numpy as np
import scipy.ndimage
from torch.nn.functional import cosine_similarity

import torch
from matplotlib import pyplot as plt


def denormalize_(image):  # 去归一化
    image = image * (255 - 0) + 0
    return image


def array2image(array):
    image = np.ceil(255 * (array - np.min(array)) / (np.max(array) - np.min(array)))
    return image


def normalize_(array):
    array = array / np.max(array)
    return array


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 0.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


def calculate_MSE(target, ref):
    H, W = target.shape[0], target.shape[1]
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    mse = ((img2 - img1) ** 2).sum() / (H * W)
    return mse


def patch2all(patch_set, numx, numy):
    # 把衍射图按照扫描位置拼接成一个大图
    positions0 = np.zeros((len(patch_set), 2), dtype=np.int32)
    positions0[:, 1] = np.tile(np.arange(numy) * patch_set.shape[1], numx)
    positions0[:, 0] = np.repeat(np.arange(numx) * patch_set.shape[2], numy)
    sample = np.zeros((numx * patch_set.shape[1], numy * patch_set.shape[2]))
    indi_x, indi_y = np.indices((patch_set.shape[1], patch_set.shape[2]))
    for i, pos in enumerate(positions0):
        sample[pos[0] + indi_x, pos[1] + indi_y] = patch_set[i, :, :]
    sample = np.array(sample)
    return sample


# 读取探针数据并将其弄到合适的大小
def probe_read(path, probe_size):
    probe = np.load(path)
    ori_shape = probe.shape
    y = (probe_size[0] - ori_shape[0]) // 2
    x = (probe_size[1] - ori_shape[1]) // 2
    probe = np.pad(probe, ((y, y), (x, x)))
    return probe


def mask(size, edge):
    '''
    generate mask to make sure target right
    parameter:
    size: the target ROI size
    edge: how many 0 pad to ROI round
    '''
    msk = np.ones(size)
    msk = np.pad(msk, ((edge, edge), (edge, edge)), constant_values=0)
    return msk


def cosine_similar(x, y):
    assert x.type() == 'torch.FloatTensor'
    assert y.type() == 'torch.FloatTensor'

    return cosine_similarity(x, y)


def checkpoint(net, opt, epoch, name, systime):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def spinavej(x):
    '''
    Read the shape and dimensions of the input image
    '''
    shape = np.shape(x)
    dim = np.size(shape)

    if dim == 2:
        nr, nc = shape
        nrdc = np.floor(nr / 2) + 1
        ncdc = np.floor(nc / 2) + 1
        r = np.arange(nr) - nrdc + 1
        c = np.arange(nc) - ncdc + 1
        [R, C] = np.meshgrid(r, c)
        index = np.round(np.sqrt(R ** 2 + C ** 2)) + 1

    elif dim == 3:
        nr, nc, nz = shape
        nrdc = np.floor(nr / 2) + 1
        ncdc = np.floor(nc / 2) + 1
        nzdc = np.floor(nz / 2) + 1
        r = np.arange(nr) - nrdc + 1
        c = np.arange(nc) - ncdc + 1
        z = np.arange(nz) - nzdc + 1
        [R, C, Z] = np.meshgrid(r, c, z)
        index = np.round(np.sqrt(R ** 2 + C ** 2 + Z ** 2)) + 1
    else:
        print('Input is neither a 2D or 3D array')

    maxindex = np.max(index)
    output = np.zeros(int(maxindex), dtype=complex)

    if nr >= 512:
        print('Performed by pixel method')
        sumf = np.zeros(int(maxindex), dtype=complex)
        count = np.zeros(int(maxindex), dtype=complex)
        for ri in range(nr):
            for ci in range(nc):
                if index[ci, ri] <= maxindex:  # 边界检查
                    sumf[int(index[ci, ri]) - 1] += x[ri, ci]
                    count[int(index[ci, ri]) - 1] += 1
        output = sumf / count
        return output
    else:
        print('Performed by index method')
        indices = [np.where(index == i + 1) for i in np.arange(int(maxindex))]
        for i in np.arange(int(maxindex)):
            output[i] = np.sum(x[indices[i]]) / len(indices[i][0])
        return output


def compute_frc(original_image, reconstructed_image):
    """
    Compute the Fourier Ring Correlation (FRC) of two images.
    Parameters
    ----------
    original_image : ndarray
        The original image.
    reconstructed_image : ndarray
        The reconstructed image.
    Returns
    -------
    frc : ndarray
        The Fourier Ring Correlation (FRC) curve.
    """
    # 将图像转换为傅里叶空间
    original_image_fft = np.fft.fftshift(np.fft.fft2(original_image))
    reconstructed_image_fft = np.fft.fftshift(np.fft.fft2(reconstructed_image))

    # 计算两个图像在傅里叶空间中的相关性
    corr = spinavej(np.multiply(original_image_fft, np.conj(reconstructed_image_fft)))
    corr1 = spinavej(np.abs(original_image_fft) ** 2)
    corr2 = spinavej(np.abs(reconstructed_image_fft) ** 2)

    # 根据FRC的定义，计算相关性的平方根，并归一化到0到1的范围
    frc = corr / np.sqrt(corr1 * corr2)
    # 把frc归一化
    frc = frc / np.max(frc)
    frc = np.clip(frc, 0, 1)
    # frc = np.where(frc == 0, 1e-6, frc)  # 将FRC曲线中小于0.2071的值设为0

    # 绘制FRC曲线，并使用半比特阈值函数来确定分辨率
    freq = np.linspace(0, 0.05, len(frc))  # 计算空间频率

    half_bit = 0.2071 + 1.9102 / freq - 7.4774 / freq ** 2 + 6.8839 / freq ** 3  # 计算半比特阈值函数

    return frc, freq


if __name__ == '__main__':
    a = np.random.rand(1606, 2355)
    b = np.random.rand(1606, 2355)
    c, d = compute_frc(a, b)

    plt.plot(d, c)
    plt.show()
