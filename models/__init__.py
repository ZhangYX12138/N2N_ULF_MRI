
from models import Noise2Noise_model
def create_model(opts):

    if opts.model_type == 'Noise2Noise_model':
        model = Noise2Noise_model.RecurrentModel(opts)

    else:
        raise NotImplementedError

    return model
