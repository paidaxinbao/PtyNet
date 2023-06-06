import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Train_DataLoader(Dataset):
    def __init__(self, data_dir, label_dir):
        super(Train_DataLoader, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.obj_pth = os.path.join(self.label_dir, 'obj_part_target_0&25&50&75.npy')

        self.train_data = np.load(self.data_dir)

        self.target_obj = np.load(self.obj_pth)
        print('fetch {} samples for training'.format(len(self.train_data)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_data[index]
        # target
        amp = np.abs(self.target_obj[index])
        phase = np.angle(self.target_obj[index])
        # np.ndarray to torch.tensor
        fn = torch.tensor(fn)
        amp = torch.tensor(amp)
        phase = torch.tensor(phase)
        return fn, amp, phase

    def __len__(self):
        return len(self.train_data)


class Test_DataLoader(Dataset):
    def __init__(self, data_dir, label_dir, opt):
        super(Test_DataLoader, self).__init__()
        self.opt = opt
        self.data_dir = data_dir
        self.label_dir = label_dir
        if self.opt.sim:
            self.amp_pth = os.path.join(self.label_dir, '1GT75overlap_newcat10_abs.npy')
            self.phase_pth = os.path.join(self.label_dir, '1GT75overlap_newcat10_angle.npy')
            self.test_data = np.load(self.data_dir).astype(np.float32)
            self.test_data = np.reshape(self.test_data, (-1, self.test_data.shape[2], self.test_data.shape[3]))
            self.target_amp = np.load(self.amp_pth)[:, 3:-3, 3:-3]
            self.target_pha = np.load(self.phase_pth)[:, 3:-3, 3:-3]
        else:
            self.amp_pth = os.path.join(self.label_dir, 'result_absFCC.npy')
            self.phase_pth = os.path.join(self.label_dir, 'result_angleFCC.npy')
            self.test_data = np.load(self.data_dir).astype(np.float32)
            self.target_amp = np.load(self.amp_pth)
            self.target_pha = np.load(self.phase_pth)

        print('fetch {} samples for testing'.format(len(self.test_data)))

    def __getitem__(self, index):
        # fetch image
        fn = self.test_data[index]
        fn = torch.tensor(fn)
        return fn

    def __len__(self):
        return len(self.test_data)
