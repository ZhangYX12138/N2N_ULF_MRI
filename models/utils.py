import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
from torch.optim import lr_scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer, opts, last_epoch=-1):
    if 'lr_policy' not in opts or opts.lr_policy == 'constant':
        scheduler = None
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.step_size,
                                        gamma=opts.gamma, last_epoch=last_epoch)
    elif opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.epoch_decay) / float(opts.n_epochs - opts.epoch_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def get_recon_loss(opts):
    loss = None
    if opts['recon'] == 'L2':
        loss = nn.MSELoss()
    elif opts['recon'] == 'L1':
        loss = nn.L1Loss()

    return loss


def psnr(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    peak_signal = (gt_image.max() - gt_image.min()).item()

    mse = (sr_image - gt_image).pow(2).mean().item()

    return 10 * log10(peak_signal ** 2 / mse)

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input 2, w, h
    output w, h, 2
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
    data = data.permute(1,2,0) #[w,h,2]

    # assert data.size(-1) == 2
    data = data
    # data = ifftshift(data, dim=(-3, -2))
    if len(data.shape) == 3:
        data_c = torch.complex(data[:, :, 0], data[:, :, 1])
    else:
        data_c = torch.complex(data[:, :, :, 0], data[:, :, :, 1])
    data = torch.fft.fft2(data_c)
    data_real = data.real
    data_imag = data.imag
    data_p = torch.stack([data_real, data_imag], dim=-1)
    # data = fftshift(data_p, dim=(-3, -2))
    data = data_p
    # data = data.permute(2,0,1) #[2,w,h]
    return data

def ifft2_data(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input 2, w, h
    output w, h, 2
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
    # data = data.permute(1,2,0) #[w,h,2]

    # assert data.size(-1) == 2
    # data = data.permute(1,2,0)
    # data = ifftshift(data, dim=(-3, -2))
    if len(data.shape) == 3:
        data_c = torch.complex(data[:, :, 0], data[:, :, 1])
    else:
        data_c = torch.complex(data[:, :, :, 0], data[:, :, :, 1])
    data = torch.fft.ifft2(data_c)
    data_real = data.real
    data_imag = data.imag
    data_p = torch.stack([data_real, data_imag], dim=-1)
    # data_p = torch.fft.ifftshift(data_p, dim=(-3, -2))
    data = data_p
    # data = data.permute(2,0,1) #[2,w,h]
    return data

def fft2_net(data, shift = True):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input b,2, w, h need to b,w, h, 2
    output b,2, w, h
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
    # data = data.permute(1,2,0) #[w,h,2]

    # assert data.size(-1) == 2
    # print(data.size())
    data = data.permute(0, 2, 3, 1) #[b,w,h,2]
    data_p = torch.zeros_like(data)
    # data = ifftshift(data, dim=(-3, -2))
    data_c = torch.complex(data[:, :, :, 0::2], data[:, :, :, 1::2])
    data = torch.fft.fftn(data_c, dim=(-3, -2, -1))
    if shift:
        data = torch.fft.fftshift(data, dim=(-3, -2, -1))
    data_real = data.real
    data_imag = data.imag
    data_p[:, :, :, 0::2] = data_real
    data_p[:, :, :, 1::2] = data_imag
    # data = fftshift(data_p, dim=(-3, -2))
    data = data_p
    data = data.permute(0, 3, 1, 2) #[b,2,w,h]
    # data = data.permute(2,0,1) #[2,w,h]
    return data


def ifft2_net(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input [b,2,w,h].
    """
    # assert data.size(-1) == 2
    data = data.permute(0,2,3,1)#[0,w,h,2]
    # data = ifftshift(data, dim=(-3, -2))
    data_p = torch.zeros_like(data)
    data_c = torch.complex(data[:, :, :, 0::2], data[:, :, :, 1::2])
    data = torch.fft.ifftn(data_c, dim=(-3, -2, -1))
    data_real = data.real
    data_imag = data.imag
    data_p[:, :, :, 0::2] = data_real
    data_p[:, :, :, 1::2] = data_imag
    # data = fftshift(data_p, dim=(-3, -2))
    data = data_p
    data = data.permute(0, 3, 1, 2)  # [0,2,w,h]
    return data


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_eval(data):
    assert data.size(1) == 2
    return (data[:, 0:1, :, :] ** 2 + data[:, 1:2, :, :] ** 2).sqrt()


def random_cropping(x, patch_size, number):
    if isinstance(x, tuple):
        if min(x[0].shape[2], x[0].shape[3]) < patch_size:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], scale_factor=0.1 + patch_size / min(x[i].shape[2], x[i].shape[3]))

        b, c, w, h = x[0].size()
        ix = np.random.choice(w - patch_size + 1, number)
        iy = np.random.choice(h - patch_size + 1, number)
        patch = [[] for _ in range(len(x))]
        for i in range(number):
            for l in range(len(x)):
                if i == 0:
                    patch[l] = x[l][:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]
                else:
                    patch[l] = torch.cat((patch[l], x[l][:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]),
                                         dim=0)
    else:
        c, w, h = x.size()

        ix = np.random.choice(w - patch_size + 1, number)
        iy = np.random.choice(h - patch_size + 1, number)

        for i in range(number):
            if i == 0:
                patch = x[:, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]
            else:
                patch = torch.cat((patch, x[:, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]), dim=0)

    return patch, ix, iy