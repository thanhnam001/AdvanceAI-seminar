from torch import optim
from lion import Lion

def sgd(model_params, lr, weight_decay=0.001, momentum=0.9):
    return optim.SGD(model_params,
                     lr=lr,
                     weight_decay=weight_decay,
                     momentum=momentum)

def adam(model_params, lr, betas=(0.9, 0.999), weight_decay=0.001):
    return optim.Adam(model_params,
                      lr=lr,
                      betas=betas,
                      weight_decay=weight_decay)

def adamw(model_params, lr, betas=(0.9, 0.999), weight_decay=0.001):
    return optim.AdamW(model_params,
                       lr=lr,
                       betas=betas,
                       weight_decay=weight_decay)

def lion(model_params, lr, betas=(0.9, 0.99), weight_decay=0.0):
    return Lion(model_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay)