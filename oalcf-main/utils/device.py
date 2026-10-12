import torch


def get_device(use_gpu=True):
    '''Resolve cuda -> mps -> cpu, honoring an explicit request to stay on CPU.'''
    if not use_gpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_accelerator(use_gpu=True):
    '''PyTorch Lightning accelerator/devices for the resolved device.'''
    device = get_device(use_gpu)
    if device.type == 'cpu':
        return 'cpu', 1
    return device.type, 1
