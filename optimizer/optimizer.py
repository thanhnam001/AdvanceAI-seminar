from torch import optim
from lion import Lion

def sgd(model_params, lr, **kwargs):
    return optim.SGD(model_params,
                     lr=lr,
                     **kwargs)

def adam(model_params, lr, **kwargs):
    return optim.Adam(model_params,
                      lr=lr,
                      **kwargs)

def adamw(model_params, lr, **kwargs):
    return optim.AdamW(model_params,
                       lr=lr,
                       **kwargs)

def lion(model_params, lr, **kwargs):
    return Lion(model_params,
                lr=lr,
                **kwargs)