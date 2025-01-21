from . import noise2noise_dataset


def get_datasets(opts):

    if opts.dataset == 'Real_noise2noise':
        trainset = noise2noise_dataset.ULFDataset_Cartesian_noise2noise(opts, mode='N2N_DEMO')
        valset = noise2noise_dataset.ULFDataset_Cartesian_noise2noise(opts, mode='N2N_DEMO')
        testset = noise2noise_dataset.ULFDataset_Cartesian_noise2noise(opts, mode='N2N_DEMO')

    else:
        raise NotImplementedError

    return trainset, valset, testset
