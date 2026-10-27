"""Only EMCift is shipped in this release."""
def get_model(model_name, args):
    if model_name.lower() != 'emcift':
        raise NotImplementedError('This repository contains EMCift only.')
    from models.emcift import Learner
    return Learner(args)
