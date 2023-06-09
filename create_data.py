import argparse
import math
import os
import random

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import torch


def image(img_path1, img_path2, obj_size):
    """
    :param img_path1: amplitude
    :param img_path2: phase
    :param obj_size: probe_size + (Nx - 1) * step_size
    """
    probe = np.load(r"pxy_probe1000.npy")
    blank = 3
    f1 = plt.imread(img_path1)
    # f1 = 0.299 * f1[:, :, 0] + 0.587 * f1[:, :, 1] + 0.114 * f1[:, :, 2] # Grayscale
    f1 = zoom(f1, (obj_size / f1.shape[0], obj_size / f1.shape[1]), order=0)
    f2 = plt.imread(img_path2)
    # f2 = 0.299 * f2[:, :, 0] + 0.587 * f2[:, :, 1] + 0.114 * f2[:, :, 2]  # Grayscale
    f2 = zoom(f2, (obj_size / f2.shape[0], obj_size / f2.shape[1]), order=0)
    f1 = f1 / np.max(f1) + 0.5
    f1 = f1 / np.max(f1)
    f1 = np.pad(f1, ((blank, blank), (blank, blank)), 'constant', constant_values=(1, 1))
    f2 = f2 / np.max(f2) + 0.5
    f2 = f2 / np.max(f2)
    f2 = np.pad(f2, ((blank, blank), (blank, blank)), 'constant', constant_values=(1, 1))

    # Randomly selects the order of amplitude and phase
    if random.random() < 0.5:
        f1 = f1
        f2 = f2
    else:
        f1 = f2
        f2 = f1

    obj = f1 * np.exp(1j * f2 * np.pi * 0.5)
    return probe, obj


def diff(probe, obj, real_positions, add_nosie: str):
    """
    To get the diffraction pattern and the target object of every scan position

    :param probe: probe, size(128, 128)
    :param obj: Simulated objects, including amplitude and phase
    :param real_positions: Scan position
    :param add_nosie: 'guassian', 'poisson', None
    :return: diff, obj_target
    """

    diff = []
    obj_target = []
    Scany, Scanx = np.indices(probe.shape)
    for pos in real_positions:
        diff_ = np.abs(fft.fftshift(fft.fft2(obj[pos[0] + Scany, pos[1] + Scanx] * probe))) ** 2
        if add_nosie == 'guassian':
            diff_ = diff_ + np.random.normal(0, 0.01, diff_.shape)
        elif add_nosie == 'poisson':
            diff_ = np.random.poisson(diff_, diff_.shape)
        elif add_nosie is None:
            diff_ = diff_
        diff.append(diff_)
        obj_target.append(obj[pos[0] + Scany, pos[1] + Scanx])
    diff = np.array(diff)
    obj_target = np.array(obj_target)
    return diff, obj_target


def position_generator(origin=3, Nx=5, Ny=5, dx=64, dy=64):
    """
    :param origin: The origin of the scan
    :param Nx: The number of scans in the x direction
    :param Ny: The number of scans in the y direction
    :param dx: The distance between two adjacent scans in the x direction
    :param dy: The distance between two adjacent scans in the y direction
    :return: real_positions
    """
    positions = np.zeros((Ny * Nx, 2), dtype=np.int32)
    positions[:, 0] = np.repeat(np.arange(Ny) * dy, Nx)
    positions[:, 1] = np.tile(np.arange(Nx) * dx, Ny)
    # Generates random jitter
    ran = np.random.randint(0, 3, size=(Nx * Ny))
    ran = ran * 1
    ran1 = np.random.randint(0, 3, size=(Nx * Ny))
    ran1 = ran1 * 1
    ran = np.array([ran1, ran])
    ran = np.transpose(ran)
    # print(ran)
    real_positions = positions + ran
    real_positions += origin
    real_positions = real_positions.astype('i4')
    return real_positions


def generate_diff_and_obj_data(data_path, num, object_size, overlap_rate, model, add_noise: str, ):
    """
    :param data_path: Image path
    :param num: Number of diffraction pattern samples to be generated by simulation
    :param object_size: Object size 608, 640, 608, 640
    :param overlap_rate: Overlap rate, selectable 75, 50, 25, 0
    :return:

    notice:
    75% overlap Nx=16, Ny=16, dx=dy=32
    50% overlap Nx=Ny=9, dx=dy=64
    25% overlap Nx=Ny=6, dx=dy=96
    0% overlap Nx=Ny=5, dx=dy=128
    """
    origin = 3
    if overlap_rate == 75:
        Nx = Ny = 16
        dx = dy = 32
        positions = position_generator(origin, Nx, Ny, dx, dy)
    elif overlap_rate == 50:
        Nx = Ny = 9
        dx = dy = 64
        positions = position_generator(origin, Nx, Ny, dx, dy)
    elif overlap_rate == 25:
        Nx = Ny = 6
        dx = dy = 96
        positions = position_generator(origin, Nx, Ny, dx, dy)
    elif overlap_rate == 0:
        Nx = Ny = 5
        dx = dy = 128
        positions = position_generator(origin, Nx, Ny, dx, dy)
    lst = os.listdir(data_path)
    diff_lst = []
    obj_lst = []
    target_lst = []
    for i in range(num):
        image1 = random.choice(lst)
        image2 = random.choice(lst)
        path1 = os.path.join(data_path, image1 if image1.endswith(('.jpg', '.png', 'jpeg')) else None)
        path2 = os.path.join(data_path, image2 if image2.endswith(('.jpg', '.png', 'jpeg')) else None)
        probe, obj = image(path1, path2, object_size)
        print('probe and object{} generated'.format(i))
        diff_, target = diff(probe, obj, positions, add_noise)
        print(f'the {i}th diffraction generated, shape {diff_.shape}, target shape {target.shape}')
        diff_lst.append(diff_)
        target_lst.append(target)
        if model == 'test':
            obj_lst.append(obj)
    diff_lst = np.array(diff_lst).astype(np.float32)
    target_lst = np.array(target_lst).astype(np.complex64)
    np.save(f'diff_cat100_{overlap_rate}overlap.npy', diff_lst)
    np.save(f'obj_part_target_{overlap_rate}overlap.npy', target_lst)
    if model == 'test':
        obj_lst = np.array(obj_lst).astype(np.complex64)
        np.save(f'GT{overlap_rate}overlap_newcat10_abs.npy', np.abs(obj_lst))
        np.save(f'GT{overlap_rate}overlap_newcat10_angle.npy', np.angle(obj_lst))

    return print('diffraction dataset 、object target and validation data have been generated')


def diff_t(probe, amp, phase, positions):
    """
    Parallelization of the generated diffraction pattern
    """
    obj = torch.complex(amp, phase)
    Scany, Scanx = torch.meshgrid(probe.shape)
    diff = []
    for pos in positions:
        diff.append(
            torch.abs(torch.fft.fftshift(
                torch.fft.fftn(obj[:, :, pos[0] + Scany, pos[1] + Scanx] * probe, dim=[-1, -2]))) ** 2)
    return torch.stack(diff, dim=0)


def concat_data():
    diff_0 = np.reshape(np.load('diff_cat100_0overlap.npy'), (-1, 128, 128))
    diff_25 = np.reshape(np.load('diff_cat100_25overlap.npy'), (-1, 128, 128))
    diff_50 = np.reshape(np.load('diff_cat100_50overlap.npy'), (-1, 128, 128))
    diff_75 = np.reshape(np.load('diff_cat100_75overlap.npy'), (-1, 128, 128))
    diff = np.concatenate((diff_0, diff_25, diff_50, diff_75), axis=0)
    np.save('diff_cat100_0&25&50&75.npy', diff)
    print('diffraction dataset have been generated')
    obj_0 = np.reshape(np.load('obj_part_target_0overlap.npy'), (-1, 128, 128))
    obj_25 = np.reshape(np.load('obj_part_target_25overlap.npy'), (-1, 128, 128))
    obj_50 = np.reshape(np.load('obj_part_target_50overlap.npy'), (-1, 128, 128))
    obj_75 = np.reshape(np.load('obj_part_target_75overlap.npy'), (-1, 128, 128))
    obj = np.concatenate((obj_0, obj_25, obj_50, obj_75), axis=0)
    np.save('obj_part_target_0&25&50&75.npy', obj)
    print('object target dataset have been generated')


if __name__ == '__main__':
    # Interactive input parameters create_data and concat_data, and become bool type
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='test')
    parser.add_argument('--data_path', type=str, default='cat_face')
    parser.add_argument('--object_num', type=int, default=1)
    parser.add_argument('--object_size', type=int, default=608)
    parser.add_argument('--overlap_rate', type=int, default=75)
    parser.add_argument('--creat_data', type=bool, default=True)
    parser.add_argument('--concat_data', type=bool, default=False)

    args = parser.parse_args()
    if args.creat_data:
        generate_diff_and_obj_data(args.data_path, args.object_num, args.object_size, args.overlap_rate, args.model,
                                   add_noise=None)

    if args.concat_data:
        concat_data()
