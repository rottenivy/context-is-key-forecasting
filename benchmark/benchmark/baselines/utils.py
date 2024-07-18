import torch


def torch_default_device():
    """
    Determine which device to use for calculations automatically.

    Notes: MPS is prioritized if available for the case where code is running on a M* MacOs device.

    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
