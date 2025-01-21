import os
import scipy.io as sio
import torch.utils.data as data
from models.utils import to_tensor
import numpy as np
import torch

class ULFDataset_Cartesian_noise2noise(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode

        if self.mode == 'N2N_DEMO':
            self.data_dir_flair = os.path.join(opts.data_root, 'N2N_DEMO')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)  # ref kspace directory (T1)


    def __getitem__(self, idx):

        slice_name = self.sample_list[idx].strip('\n')
        ULF_IMG = sio.loadmat(os.path.join(self.data_dir_flair, slice_name))

        ULF_IMG_1_1 = np.concatenate([ULF_IMG['S1_1'].real[np.newaxis, :, :], ULF_IMG['S1_1'].imag[np.newaxis, :, :]], axis=0)
        ULF_IMG_1_2 = np.concatenate([ULF_IMG['S1_2'].real[np.newaxis, :, :], ULF_IMG['S1_2'].imag[np.newaxis, :, :]], axis=0)
        ULF_IMG_1_3 = np.concatenate([ULF_IMG['S1_3'].real[np.newaxis, :, :], ULF_IMG['S1_3'].imag[np.newaxis, :, :]], axis=0)

        ULF_IMG_2_1 = np.concatenate([ULF_IMG['S2_1'].real[np.newaxis, :, :], ULF_IMG['S2_1'].imag[np.newaxis, :, :]], axis=0)
        ULF_IMG_2_2 = np.concatenate([ULF_IMG['S2_2'].real[np.newaxis, :, :], ULF_IMG['S2_2'].imag[np.newaxis, :, :]], axis=0)
        ULF_IMG_2_3 = np.concatenate([ULF_IMG['S2_3'].real[np.newaxis, :, :], ULF_IMG['S2_3'].imag[np.newaxis, :, :]], axis=0)

        ULF_IMG_1 = np.concatenate([ULF_IMG_1_1, ULF_IMG_1_2, ULF_IMG_1_3], axis=0, dtype=float)
        ULF_IMG_2 = np.concatenate([ULF_IMG_2_1, ULF_IMG_2_2, ULF_IMG_2_3], axis=0, dtype=float)

        ULF_IMG_AVG = (ULF_IMG_1 + ULF_IMG_2)/2

        std_mea = ULF_IMG_1[:, 0:8, 0:8]
        sigma1 = np.std(std_mea[:])
        std_mea = ULF_IMG_2[:, 0:8, 0:8]
        sigma2 = np.std(std_mea[:])
        std_mea = ULF_IMG_AVG[:, 0:8, 0:8]
        sigma_avg = np.std(std_mea[:])

        source = to_tensor(ULF_IMG_1).float()
        target = to_tensor(ULF_IMG_2).float()
        im_avg = to_tensor(ULF_IMG_AVG).float()

        sigma_mat = to_tensor(np.ones((1, source.shape[1], source.shape[2]))).float()

        lamda = 1.0

        source_plus = torch.cat([source, sigma_mat*lamda*(sigma1+sigma2)/2], dim=0)
        target_plus = torch.cat([target, sigma_mat*lamda*(sigma1+sigma2)/2], dim=0)
        avg_plus = torch.cat([im_avg, sigma_mat*lamda*sigma_avg], dim=0)

        return {'image_full': target,
                'image_sub_1': source_plus,
                'image_sub_2': target_plus,
                'image_sub_avg': avg_plus,
                }

    def __len__(self):
        return len(self.sample_list)

class ULFDataset_Simulation_noise2noise(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode

        if self.mode == 'N2N_DEMO':
            self.data_dir_flair = os.path.join(opts.data_root, 'N2N_DEMO')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)  # ref kspace directory (T1)


    def __getitem__(self, idx):

        slice_name = self.sample_list[idx].strip('\n')
        SIMU_IMG = sio.loadmat(os.path.join(self.data_dir_flair, slice_name))

        SIMU_IMG_1_1 = np.concatenate([SIMU_IMG['S1_1'].real[np.newaxis, :, :], SIMU_IMG['S1_1'].imag[np.newaxis, :, :]], axis=0)
        SIMU_IMG_1_2 = np.concatenate([SIMU_IMG['S1_2'].real[np.newaxis, :, :], SIMU_IMG['S1_2'].imag[np.newaxis, :, :]], axis=0)
        SIMU_IMG_1_3 = np.concatenate([SIMU_IMG['S1_3'].real[np.newaxis, :, :], SIMU_IMG['S1_3'].imag[np.newaxis, :, :]], axis=0)

        SIMU_IMG_1 = to_tensor(np.concatenate([SIMU_IMG_1_1, SIMU_IMG_1_2, SIMU_IMG_1_3], axis=0)).float()

        sigma = 0.15

        # =======
        torch.manual_seed(idx)
        noise1 = torch.randn_like(SIMU_IMG_1) * sigma  # 生成与 k 空间数据相同大小的高斯噪声
        torch.manual_seed(idx+1)
        noise2 = torch.randn_like(SIMU_IMG_1) * sigma  # 生成与 k 空间数据相同大小的高斯噪声
        torch.seed()



        source = noise1 + SIMU_IMG_1
        target = noise2 + SIMU_IMG_1
        im_avg = (target+source)/2


        std_mea = source[:, 0:8, 0:8]
        sigma1 = torch.std(std_mea[:])
        std_mea = target[:, 0:8, 0:8]
        sigma2 = torch.std(std_mea[:])
        std_mea = im_avg[:, 0:8, 0:8]
        sigma_avg = torch.std(std_mea[:])

        sigma_mat = to_tensor(np.ones((1, source.shape[1], source.shape[2])) * np.array((sigma1 + sigma2)/2)).float()
        sigma_mat_avg = to_tensor(np.ones((1, source.shape[1], source.shape[2])) * np.array(sigma_avg)).float()

        source_plus = torch.cat([source, sigma_mat], dim=0)
        target_plus = torch.cat([target, sigma_mat], dim=0)
        avg_plus = torch.cat([im_avg, sigma_mat_avg], dim=0)


        # ---------------------over------
        return {
                'image_full': SIMU_IMG_1,
                'image_sub_1': source_plus,
                'image_sub_2': target_plus,
                'image_sub_avg': avg_plus,
        }


    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    a = 1
