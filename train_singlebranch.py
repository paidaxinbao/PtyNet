from __future__ import division

import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataload import Train_DataLoader, Test_DataLoader
from Net1_3 import Network
from utils import calculate_ssim, calculate_psnr, array2image, calculate_MSE, normalize_, checkpoint, compute_frc, cosine_similar
from tqdm import tqdm


def seed_torch(seed=101):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def _init_net_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        if self.opt.model == 'train':
            self.train_dataset = Train_DataLoader(self.opt.data_dir, self.opt.target_path)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           num_workers=0,
                                           batch_size=self.opt.batchsize,
                                           shuffle=True,
                                           pin_memory=False,
                                           drop_last=True)  # 内存充足的时候，可以设置pin_memory=True，加快GPU读取数据
        else:
            self.test_dataset = Test_DataLoader(self.opt.val_dir, self.opt.val_target_dir, self.opt)
            ft_batch_size = 16 if (
                    self.opt.model == 'finetune' and self.opt.sim) else 64
            self.test_loader = DataLoader(dataset=self.test_dataset,
                                          num_workers=0,
                                          batch_size=ft_batch_size,
                                          shuffle=False,
                                          pin_memory=False,
                                          drop_last=False)

        self.network = Network()
        if self.opt.parallel:
            self.network = torch.nn.DataParallel(self.network)
        self.network = self.network.cuda()

        if self.opt.model == 'train':
            # about training scheme
            self.num_epoch = self.opt.n_epoch
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lr)
            # 循环学习率
            # Optimizer details
            iterations_per_epoch = np.floor(
                self.train_dataset.__len__() / self.opt.batchsize) + 1  # Final batch will be less than batch size
            step_size = 60 * iterations_per_epoch  # Paper recommends 2-10 number of iterations, step_size is half cycle
            print("LR step size is:", step_size, "which is every %d epochs" % (step_size / iterations_per_epoch))
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.opt.lr / 10,
                                                               max_lr=self.opt.lr, step_size_up=step_size,
                                                               cycle_momentum=False, mode='triangular2')
            print("Training with Cyclic LR")
            print("number of epoch={}".format(self.opt.n_epoch))
        elif self.opt.model == 'finetune':
            # about training scheme
            self.num_epoch = self.opt.n_epoch
            ratio = self.num_epoch / 100
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lr)
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                      milestones=[
                                                          int(20 * ratio) - 1,
                                                          int(40 * ratio) - 1,
                                                          int(60 * ratio) - 1,
                                                          int(80 * ratio) - 1
                                                      ],
                                                      gamma=self.opt.gamma)
            print("number of epoch={}".format(self.opt.n_epoch))

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()

    def train(self):
        systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        scaler = torch.cuda.amp.GradScaler()

        _init_net_weights(self.network)
        print('init network finished')

        Loss = []
        for epoch in range(1, self.opt.n_epoch + 1):
            st = time.time()
            train_bar = tqdm(self.train_loader)
            running_results = {'batch_sizes': 0, 'loss': 0, 'loss_I2': 0, 'Time': 0}

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

            self.network.train()
            for diff, amp, phase in train_bar:
                batch_size = diff.size(0)
                running_results['batch_sizes'] += batch_size

                diff = diff.cuda().float().unsqueeze(1)
                if self.opt.noisetype == 'poisson':
                    diff = torch.poisson(diff)
                elif self.opt.noisetype == 'gaussian':
                    diff = torch.normal(diff, 1)
                elif self.opt.noisetype == 'None':
                    diff = diff
                else:
                    raise ValueError('Noisetype must be poisson or gaussian or None')

                amp = amp.cuda().float().unsqueeze(1)
                phase = phase.cuda().float().unsqueeze(1)

                # normalize the amplitude
                amplitude = torch.sqrt(diff)
                max_ = torch.amax(amplitude, dim=[-2, -1], keepdim=True)
                amplitude_ = torch.div(amplitude, max_ + 1e-6)

                self.optimizer.zero_grad()
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    pred_amp, pred_pha, _ = self.network(amplitude_)

                    pred_amp = pred_amp.unsqueeze(1)
                    pred_pha = pred_pha.unsqueeze(1)
                    loss = self.criterion1(pred_amp, amp) + self.criterion1(pred_pha, phase)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()

                    # loss for current batch before optimization
                    running_results['loss'] += loss.item() * batch_size
                    running_results['Time'] = time.time() - st
                    train_bar.set_description(desc='%d Loss:%.4f'
                                                   'Time:%.4f' % (
                                                       epoch,
                                                       running_results['loss'] / running_results['batch_sizes'],
                                                       running_results['Time']))

            # save Loss
            Loss.append(running_results['loss'] / running_results['batch_sizes'])
            # save checkpoint
            if epoch % 50 == 0:
                checkpoint(self.network, self.opt, epoch, "model", systime)

        np.save('statistics/loss_results.npy', Loss)

    def validation(self):
        # validation
        # load network
        self.network.load_state_dict(torch.load(self.opt.pre_trained_model))

        # initialize all kinds of thing
        complex_objs = []
        abs_ = []
        abs_apart = []
        angle_ = []
        angle_apart = []
        amp = self.test_dataset.target_amp
        phase = self.test_dataset.target_pha

        if self.opt.sim:
            # ------------------------------------------------
            # define scanning positions in terms of pixels
            # ------------------------------------------------
            # initial position of 0
            origin = 0
            Ny = self.opt.num_diff[0]  # number of y scan
            Nx = self.opt.num_diff[0]
            dy = self.opt.stepsize
            dx = dy  # same
            positions = np.zeros((Ny * Nx, 2), dtype=np.int32)
            # y positions in 2st colum
            positions[:, 1] = np.tile(np.arange(Nx) * dx, Ny)
            # x positions in 1st colum
            positions[:, 0] = np.repeat(np.arange(Ny) * dy, Nx)
            positions += origin

            obj_indy, obj_indx = np.indices(self.opt.probe_size)

        else:
            positions = np.load('./train/train_data/FCC/FCC_position.npy')
            obj_indy, obj_indx = np.indices(self.opt.probe_size)

        print('Scanning positions defined')
        with torch.no_grad():
            for num, valid_data in enumerate(self.test_loader):
                num += 1
                valid_data = valid_data.cuda().float().unsqueeze(1)
                amplitude_ = torch.sqrt(valid_data)
                # normalize the amplitude
                max_diff = torch.amax(amplitude_, dim=[-1, -2], keepdim=True)
                amplitude_ = torch.div(amplitude_, max_diff + 1e-6)
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    pred_amp, pred_pha, obj = self.network(amplitude_)
                    complex_obj = obj
                    complex_objs.append(complex_obj)
            complex_objs = torch.cat(complex_objs, dim=0).cpu().numpy().squeeze()
            if self.opt.sim:
                obj_Func = np.zeros(self.opt.scan_area, dtype=np.complex64)
                count_matrix = np.zeros(self.opt.scan_area)  # Counting matrix, used to count the scanned positions, the area scanned is increased by 1
                # Stitching
                num_ = self.test_dataset.__len__() // (Nx * Ny)
                count_probe = np.ones((Ny * Nx, self.opt.probe_size[0], self.opt.probe_size[1]))
                for j in range(num_):
                    for i, pos in enumerate(positions):
                        obj_Func[pos[0] + obj_indy, pos[1] + obj_indx] += complex_objs[i + (Nx * Ny * j)]
                        count_matrix[pos[0] + obj_indy, pos[1] + obj_indx] += count_probe[i]

                        abs_apart.append(np.abs(complex_objs[i + (Nx * Ny * j)]))
                        angle_apart.append(np.angle(complex_objs[i + (Nx * Ny * j)]))

                    obj_Func = obj_Func / count_matrix

                    abs_.append(np.abs(obj_Func))
                    angle_.append(np.angle(obj_Func))

                    # calculate the psnr and ssim
                    psnr_amp = calculate_psnr(array2image(np.abs(obj_Func)), array2image(amp[j]))
                    psnr_angle = calculate_psnr(array2image(np.angle(obj_Func)), array2image(phase[j]))
                    ssim_amp = calculate_ssim(array2image(np.abs(obj_Func)), array2image(amp[j]))
                    ssim_angle = calculate_ssim(array2image(np.angle(obj_Func)), array2image(phase[j]))
                    mse_amp = calculate_MSE(normalize_(np.abs(obj_Func)), normalize_(amp[j]))
                    mse_angle = calculate_MSE(normalize_(np.angle(obj_Func)), normalize_(phase[j]))

                    # visualization
                    fig, axes = plt.subplots(2, 2)

                    a0 = axes[0, 0].imshow(np.abs(obj_Func))
                    fig.colorbar(a0, ax=axes[0, 0])
                    axes[0, 0].set_title('pred amplitude, PSNR:{:.3f}, SSIM:{:.3f}, MSE:{:.3f}'.format(psnr_amp, ssim_amp, mse_amp))
                    a1 = axes[0, 1].imshow(amp[j])
                    fig.colorbar(a1, ax=axes[0, 1])
                    axes[0, 1].set_title('real amplitude')
                    a2 = axes[1, 0].imshow(np.angle(obj_Func))
                    fig.colorbar(a2, ax=axes[1, 0])
                    axes[1, 0].set_title('pred phase, PSNR:{:.3f}, SSIM:{:.3f}, MSE:{:.3f}'.format(psnr_angle, ssim_angle, mse_angle))
                    a3 = axes[1, 1].imshow(phase[j])
                    fig.colorbar(a3, ax=axes[1, 1])
                    axes[1, 1].set_title('real phase')
                    plt.show()

                # abs_, angle_ = np.asarray(abs_), np.asarray(angle_)
                # np.save(r"./results/abs_overlap25_10.npy", abs_)
                # np.save(r"./results/angle_overlap25_10.npy", angle_)
                # np.save(r"./results/abs_overlap25_10_apart.npy", abs_apart)
                # np.save(r"./results/angle_overlap25_10_apart.npy", angle_apart)

            else:
                obj_Func = np.zeros(self.opt.scan_area, dtype=np.complex64)
                ture_amp = amp
                ture_angle = phase
                count_matrix = np.zeros(self.opt.scan_area)
                count_probe = np.ones((self.opt.probe_size[0], self.opt.probe_size[1]))
                for i, pos in enumerate(positions):
                    obj_Func[pos[0] + obj_indy, pos[1] + obj_indx] += complex_objs[i]
                    count_matrix[pos[0] + obj_indy, pos[1] + obj_indx] += count_probe

                    # abs_apart.append(np.abs(complex_objs[i]))
                    # angle_apart.append(np.angle(complex_objs[i]))
                count_matrix = np.where(count_matrix == 0, 1, count_matrix)
                obj_Func = obj_Func / count_matrix
                abs_.append(np.abs(obj_Func))
                angle_.append(np.angle(obj_Func))

                # np.save(r"./results/FCC_result_abs.npy", abs_)
                # np.save(r"./results/FCC_result_angle.npy", angle_)
                # np.save(r"./results/FCC_ture_abs.npy", ture_amp)
                # np.save(r"./results/FCC_ture_angle.npy", ture_angle)

                psnr_amp = calculate_psnr(array2image(np.abs(obj_Func)), array2image(ture_amp))
                psnr_angle = calculate_psnr(array2image(np.angle(obj_Func)), array2image(ture_angle))
                ssim_amp = calculate_ssim(array2image(np.abs(obj_Func)), array2image(ture_amp))
                ssim_angle = calculate_ssim(array2image(np.angle(obj_Func)), array2image(ture_angle))
                mse_amp = calculate_MSE(normalize_(np.abs(obj_Func)), normalize_(ture_amp))
                mse_angle = calculate_MSE(normalize_(np.angle(obj_Func)), normalize_(ture_angle))

                fig, axes = plt.subplots(2, 2)

                a0 = axes[0, 0].imshow(np.abs(obj_Func), cmap='gray')
                fig.colorbar(a0, ax=axes[0, 0])
                axes[0, 0].set_title('pred amplitude, PSNR:{:.3f}, SSIM:{:.3f}, MSE:{:.3f}'.format(psnr_amp, ssim_amp, mse_amp))
                a1 = axes[0, 1].imshow(ture_amp)
                fig.colorbar(a1, ax=axes[0, 1])
                axes[0, 1].set_title('real amplitude')
                a2 = axes[1, 0].imshow(np.angle(obj_Func), cmap='gray')
                fig.colorbar(a2, ax=axes[1, 0])
                axes[1, 0].set_title('pred phase, PSNR:{:.3f}, SSIM:{:.3f}, MSE:{:.3f}'.format(psnr_angle, ssim_angle, mse_angle))
                a3 = axes[1, 1].imshow(ture_angle)
                fig.colorbar(a3, ax=axes[1, 1])
                axes[1, 1].set_title('real phase')
                plt.show()

    def finetune(self):
        scaler = torch.cuda.amp.GradScaler()
        if self.opt.sim:
            probe = np.load(r"pxy_probe1000.npy")
            # Copying the probe into batchsize copies
            probe = torch.tensor(probe).cuda().unsqueeze(0)
        #else:
            # probe = np.load('./test/FCC_data/result_probeFCC.npy')
            # probe = torch.tensor(probe).cuda().unsqueeze(0)

        self.network.load_state_dict(torch.load(self.opt.pre_trained_model))
        print('load model finished')

        Loss = []
        for epoch in range(1, self.opt.n_epoch + 1):
            st = time.time()
            train_bar = tqdm(self.test_loader)
            running_results = {'batch_sizes': 0, 'loss_I': 0, 'Time': 0}

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

            self.network.train()

            for diff in train_bar:
                batch_size = diff.size(0)
                # probe_ = probe.detach().repeat(batch_size, 1, 1, 1)
                running_results['batch_sizes'] += batch_size

                diff = diff.cuda().float().unsqueeze(1)

                # norm
                amplitude = torch.sqrt(diff)
                max_ = torch.amax(amplitude, dim=[-2, -1], keepdim=True)
                amplitude_ = torch.div(amplitude, max_ + 1e-6)

                self.optimizer.zero_grad()

                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    #     # Freeze the first two layers of encoder and decoder if u want speed up(but may cause performance drop)
                    #     for param in self.network.encoder.parameters():
                    #         param.requires_grad = False
                    #     for param in self.network.decoder[0:2].parameters():
                    #         param.requires_grad = False

                    pred_amp, pred_pha, obj = self.network(amplitude_)
                    obj = obj.unsqueeze(1)

                    distribution = torch.fft.fftshift(torch.fft.fftn(obj * probe, dim=(-2, -1)), dim=(-2, -1))
                    diff_pred = torch.abs(distribution) ** 2

                    loss_I = self.criterion2(diff, diff_pred)

                    scaler.scale(loss_I).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    # loss for current batch before optimization
                    running_results['loss_I'] += loss_I.item() * batch_size
                    running_results['Time'] = time.time() - st
                    train_bar.set_description(desc='%d Loss_I:%.4f '
                                                   'Time:%.4f' % (
                                                       epoch,
                                                       running_results['loss_I'] / running_results['batch_sizes'],
                                                       running_results['Time']))
            self.scheduler.step()
            # save loss
            Loss.append(running_results['loss_I'] / running_results['batch_sizes'])
        # save model
        torch.save(self.network.state_dict(), self.opt.save_finetunemodel_path)

        print('------------------------------------------------------------')
        print('Finished finetune, start testing')
        print('------------------------------------------------------------')

        self.network.eval()

        # initialize all kinds of thing
        abs_ = []
        abs_apart = []
        angle_ = []
        angle_apart = []
        amp = self.test_dataset.target_amp
        phase = self.test_dataset.target_pha

        if self.opt.sim:
            # ------------------------------------------------
            # define scanning positions in terms of pixels
            # ------------------------------------------------
            origin = 0
            Ny = self.opt.num_diff[0]
            Nx = self.opt.num_diff[1]
            dy = self.opt.stepsize
            dx = dy  # same
            positions = np.zeros((Ny * Nx, 2), dtype=np.int32)
            # y positions in 2st colum
            positions[:, 1] = np.tile(np.arange(Nx) * dx, Ny)
            # x positions in 1st colum
            positions[:, 0] = np.repeat(np.arange(Ny) * dy, Nx)
            positions += origin

            obj_indy, obj_indx = np.indices(self.opt.probe_size)

        else:
            positions = np.load('./test/FCC_data/FCC_position.npy')
            obj_indy, obj_indx = np.indices(self.opt.probe_size)

        print('Scanning positions defined')
        with torch.no_grad():
            obj_set = []
            for num, valid_data in enumerate(self.test_loader):
                num += 1
                valid_data = valid_data.cuda().float().unsqueeze(1)
                valid_data = torch.sqrt(valid_data)
                max_diff = torch.amax(valid_data, dim=[-1, -2], keepdim=True)
                diff = torch.div(valid_data, max_diff + 1e-6)
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    pred_amp, pred_pha, obj = self.network(diff)
                    complex_obj_set = obj
                    if not self.opt.sim:
                        obj_set.append(complex_obj_set)
            obj_set = torch.concat(obj_set, axis=0).squeeze().cpu().numpy()
            if self.opt.sim:
                obj_Func = np.zeros(self.opt.scan_area, dtype=np.complex64)
                count_matrix = np.zeros(self.opt.scan_area)
                # Stitching
                num_ = self.test_dataset.__len__() // (Nx * Ny)
                count_probe = np.ones((self.opt.probe_size[0], self.opt.probe_size[1]))
                for j in range(num_):
                    for i, pos in enumerate(positions):
                        obj_Func[pos[0] + obj_indy, pos[1] + obj_indx] += complex_obj_set[i + (Nx * Ny * j)]
                        count_matrix[pos[0] + obj_indy, pos[1] + obj_indx] += count_probe

                        abs_apart.append(np.abs(complex_obj_set[i + (Nx * Ny * j)]))
                        angle_apart.append(np.angle(complex_obj_set[i + (Nx * Ny * j)]))

                    obj_Func = obj_Func / count_matrix

                    abs_.append(np.abs(obj_Func))
                    angle_.append(np.angle(obj_Func))

                    # calculate the psnr, ssim, frc

                    psnr_amp = calculate_psnr(array2image(np.abs(obj_Func)), array2image(amp[j]))
                    psnr_angle = calculate_psnr(array2image(np.angle(obj_Func)), array2image(phase[j]))
                    ssim_amp = calculate_ssim(array2image(np.abs(obj_Func)), array2image(amp[j]))
                    ssim_angle = calculate_ssim(array2image(np.angle(obj_Func)), array2image(phase[j]))
                    mse_amp = calculate_MSE(normalize_(np.abs(obj_Func)), normalize_(amp[j]))
                    mse_angle = calculate_MSE(normalize_(np.angle(obj_Func)), normalize_(phase[j]))
                    frc_amp, freq_amp = compute_frc(amp[j], np.abs(obj_Func))
                    frc_angle, freq_angle = compute_frc(phase[j], np.angle(obj_Func))

                    error_amp = np.std(frc_amp) / np.sqrt(frc_amp.size)
                    error_angle = np.std(frc_angle) / np.sqrt(frc_angle.size)
                    model1 = make_interp_spline(freq_amp, frc_amp)
                    model2 = make_interp_spline(freq_angle, frc_angle)

                    freq_amp = np.linspace(freq_amp.min(), freq_amp.max(), 800)
                    frc_amp = model1(freq_amp)
                    freq_angle = np.linspace(freq_angle.min(), freq_angle.max(), 800)
                    frc_angle = model2(freq_angle)

                    fig, axes = plt.subplots(2, 3)

                    a0 = axes[0, 0].imshow(amp[j])
                    fig.colorbar(a0, ax=axes[0, 0])
                    axes[0, 0].set_ylabel('Real Amplitude', rotation=90, labelpad=20, fontsize=14)

                    a1 = axes[1, 0].imshow(phase[j])
                    fig.colorbar(a1, ax=axes[1, 0])
                    axes[1, 0].set_ylabel('Real Phase', rotation=90, labelpad=20, fontsize=14)

                    a2 = axes[0, 1].imshow(np.abs(obj_Func))
                    fig.colorbar(a2, ax=axes[0, 1])
                    axes[0, 1].set_ylabel('Predicted Amplitude', rotation=90, labelpad=20, fontsize=14)
                    axes[0, 1].set_title(
                        'PSNR: {:.2f}  SSIM: {:.2f}  MSE: {:.2f}'.format(psnr_amp, ssim_amp,
                                                                         mse_amp))

                    a3 = axes[1, 1].imshow(np.angle(obj_Func))
                    fig.colorbar(a3, ax=axes[1, 1])
                    axes[1, 1].set_ylabel('Predicted Phase', rotation=90, labelpad=20, fontsize=14)
                    axes[1, 1].set_title(
                        'PSNR: {:.2f}  SSIM: {:.2f}  MSE: {:.2f}'.format(psnr_angle, ssim_angle,
                                                                         mse_angle))

                    # Plotting FRC curves
                    half_bit_amp = np.full(freq_amp.shape, 0.5)
                    axes[0, 2].plot(freq_amp, frc_amp, label='Amplitude')
                    axes[0, 2].fill_between(freq_amp, frc_amp - error_amp, frc_amp + error_amp, alpha=0.2)
                    axes[0, 2].plot(freq_amp, half_bit_amp, label='0.5 threshold')
                    axes[0, 2].set_xlabel("Spatial frequency (nm$^{-1}$)")
                    axes[0, 2].set_ylabel('FRC value correlation')
                    axes[0, 2].set_title('FRC - Amplitude')
                    axes[0, 2].legend()
                    axes[0, 2].grid(True)

                    # Find the intersection of the FRC curve and the half-bit threshold function, and display the resolution result
                    idx = np.argwhere(np.diff(np.sign(frc_amp - half_bit_amp))).flatten()
                    if len(idx) > 0:
                        res_freq = freq_amp[idx[0]]
                        res_dist = 1 / res_freq / 2
                        axes[0, 2].plot(res_freq, frc_amp[idx[0]], "ro")
                        axes[0, 2].text(res_freq, frc_amp[idx[0]],
                                        f"({res_freq:.3f}, {frc_amp[idx[0]]:.3f})")
                        print(f"The resolution is {res_dist:.3f} pixels.")
                    else:
                        print("The resolution cannot be determined.")

                    half_bit_angle = np.full(freq_angle.shape, 0.5)
                    axes[1, 2].plot(freq_angle, frc_angle, label='Phase')
                    axes[1, 2].fill_between(freq_angle, frc_angle - error_angle, frc_angle + error_angle, alpha=0.2)
                    axes[1, 2].plot(freq_angle, half_bit_angle, label='0.5 threshold')
                    axes[1, 2].set_xlabel('Spatial frequency (nm$^{-1}$)')
                    axes[1, 2].set_ylabel('FRC value correlation')
                    axes[1, 2].set_title('FRC - Phase')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True)

                    idx = np.argwhere(np.diff(np.sign(frc_angle - half_bit_angle))).flatten()
                    if len(idx) > 0:
                        res_freq = freq_angle[idx[0]]
                        res_dist = 1 / res_freq / 2
                        axes[1, 2].plot(res_freq, frc_angle[idx[0]], "ro")
                        axes[1, 2].text(res_freq, frc_angle[idx[0]],
                                        f"({res_freq:.3f}, {frc_angle[idx[0]]:.3f})")
                        print(f"The resolution is {res_dist:.3f} pixels.")
                    else:
                        print("The resolution cannot be determined.")

                    fig.tight_layout()

                    plt.show()

                # abs_, angle_ = np.asarray(abs_), np.asarray(angle_)
                # np.save(r"./results/abs_overlap25_10.npy", abs_)
                # np.save(r"./results/angle_overlap25_10.npy", angle_)
                # np.save(r"./results/abs_overlap25_10_apart.npy", abs_apart)
                # np.save(r"./results/angle_overlap25_10_apart.npy", angle_apart)

            else:
                obj_Func = np.zeros(self.opt.scan_area, dtype=np.complex64)
                ture_amp = amp
                ture_angle = phase
                count_matrix = np.zeros(self.opt.scan_area)
                count_probe = np.ones((self.opt.probe_size[0], self.opt.probe_size[1]))
                for i, pos in enumerate(positions):
                    obj_Func[pos[0] + obj_indy, pos[1] + obj_indx] += obj_set[i]
                    count_matrix[pos[0] + obj_indy, pos[1] + obj_indx] += count_probe

                    # abs_apart.append(np.abs(complex_objs[i]))
                    # angle_apart.append(np.angle(complex_objs[i]))
                count_matrix = np.where(count_matrix == 0, 1, count_matrix)
                obj_Func = obj_Func / count_matrix
                abs_.append(np.abs(obj_Func))
                angle_.append(np.angle(obj_Func))

                # np.save(r"./results/FCC_result_abs.npy", abs_)
                # np.save(r"./results/FCC_result_angle.npy", angle_)
                # np.save(r"./results/FCC_ture_abs.npy", ture_amp)
                # np.save(r"./results/FCC_ture_angle.npy", ture_angle)

                psnr_amp = calculate_psnr(array2image(np.abs(obj_Func)), array2image(ture_amp))
                psnr_angle = calculate_psnr(array2image(np.angle(obj_Func)), array2image(ture_angle))
                ssim_amp = calculate_ssim(array2image(np.abs(obj_Func)), array2image(ture_amp))
                ssim_angle = calculate_ssim(array2image(np.angle(obj_Func)), array2image(ture_angle))
                mse_amp = calculate_MSE(normalize_(np.abs(obj_Func)), normalize_(ture_amp))
                mse_angle = calculate_MSE(normalize_(np.angle(obj_Func)), normalize_(ture_angle))
                frc_amp, freq_amp = compute_frc(ture_amp, np.abs(obj_Func))
                frc_angle, freq_angle = compute_frc(ture_angle, np.angle(obj_Func))

                fig, axes = plt.subplots(2, 3)

                a0 = axes[0, 0].imshow(amp)
                fig.colorbar(a0, ax=axes[0, 0])
                axes[0, 0].set_ylabel('Real Amplitude', rotation=90, labelpad=20, fontsize=14)

                a1 = axes[1, 0].imshow(phase)
                fig.colorbar(a1, ax=axes[1, 0])
                axes[1, 0].set_ylabel('Real Phase', rotation=90, labelpad=20, fontsize=14)

                a2 = axes[0, 1].imshow(np.abs(obj_Func))
                fig.colorbar(a2, ax=axes[0, 1])
                axes[0, 1].set_ylabel('Predicted Amplitude', rotation=90, labelpad=20, fontsize=14)
                axes[0, 1].set_title(
                    'PSNR: {:.2f}  SSIM: {:.2f}  MSE: {:.2f}'.format(psnr_amp, ssim_amp,
                                                                     mse_amp))

                a3 = axes[1, 1].imshow(np.angle(obj_Func))
                fig.colorbar(a3, ax=axes[1, 1])
                axes[1, 1].set_ylabel('Predicted Phase', rotation=90, labelpad=20, fontsize=14)
                axes[1, 1].set_title(
                    'PSNR: {:.2f}  SSIM: {:.2f}  MSE: {:.2f}'.format(psnr_angle, ssim_angle,
                                                                     mse_angle))

                half_bit_amp = np.full(freq_amp.shape, 0.5)
                axes[0, 2].plot(freq_amp, frc_amp, label='Amplitude')
                axes[0, 2].plot(freq_amp, half_bit_amp, label='0.5 threshold')
                axes[0, 2].set_xlabel("Spatial frequency (nm$^{-1}$)")
                axes[0, 2].set_ylabel('FRC value correlation')
                axes[0, 2].set_title('FRC - Amplitude')
                axes[0, 2].legend()
                axes[0, 2].grid(True)

                idx_amp = np.argwhere(np.diff(np.sign(frc_amp - half_bit_amp))).flatten()
                if len(idx_amp) > 0:
                    res_freq = freq_amp[idx_amp[0]]
                    res_dist = 1 / res_freq / 2
                    axes[0, 2].plot(res_freq, frc_amp[idx_amp[0]], "ro")
                    axes[0, 2].text(res_freq, frc_amp[idx_amp[0]],
                                    f"({res_freq:.3f}, {frc_amp[idx_amp[0]]:.3f})")
                    print(f"The resolution is {res_dist:.3f} pixels.")
                else:
                    print("The resolution cannot be determined.")

                half_bit_angle = np.full(freq_angle.shape, 0.5)
                axes[1, 2].plot(freq_angle, frc_angle, label='Phase')
                axes[1, 2].plot(freq_angle, half_bit_angle, label='0.5 threshold')
                axes[1, 2].set_xlabel('Spatial frequency (nm$^{-1}$)')
                axes[1, 2].set_ylabel('FRC value correlation')
                axes[1, 2].set_title('FRC - Phase')
                axes[1, 2].legend()
                axes[1, 2].grid(True)

                idx_angle = np.argwhere(np.diff(np.sign(frc_angle - half_bit_angle))).flatten()
                if len(idx_angle) > 0:
                    res_freq = freq_angle[idx_angle[0]]
                    res_dist = 1 / res_freq / 2
                    axes[1, 2].plot(res_freq, frc_angle[idx_angle[0]], "ro")
                    axes[1, 2].text(res_freq, frc_angle[idx_angle[0]],
                                    f"({res_freq:.3f}, {frc_angle[idx_angle[0]]:.3f})")
                    print(f"The resolution is {res_dist:.3f} pixels.")
                else:
                    print("The resolution cannot be determined.")

                fig.tight_layout()

                plt.show()
