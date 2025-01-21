import torch.nn as nn
from networks.unet import *


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network

def get_network(opts):

    if opts.net_G == 'noise2noise_unet_3':
        network = UNet(in_channels=6+1, out_channels=6)

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)
