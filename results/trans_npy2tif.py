import os

import numpy as np
from PIL import Image

abs = np.load('abs_trainSR.npy')
angle = np.load('angle_trainSR.npy')
save_path = 'train_cat_tif'

if os.path.exists(save_path):
    print('save path existed')
else:
    os.makedirs(save_path)


def transform_val(array):
    array = 255 * (array - np.amin(array)) / (np.amax(array) - np.amin(array))
    return array


for i in range(len(abs)):
    abs_ = Image.fromarray(np.ceil(transform_val(abs[i]))).convert('L')
    angle_ = Image.fromarray(np.ceil(transform_val(angle[i]))).convert('L')
    abs_.save(os.path.join(save_path, 'train_abs{}.tif').format(i))
    angle_.save(os.path.join(save_path, 'train_angle{}.tif').format(i + 50))
