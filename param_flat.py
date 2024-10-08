import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnext50_32x4d
from tqdm import tqdm

def flatten_parameters(params):
    """Flatten model parameters into a single 1D tensor."""
    return torch.cat([param.flatten() for param in params])

def unflatten_parameters(flat_params, param_shapes):
    """Unflatten a 1D tensor into model parameters with the original shapes."""
    unflattened_params = []
    start_idx = 0
    for shape in param_shapes:
        numel = torch.prod(torch.tensor(shape)).item()
        unflattened_params.append(flat_params[start_idx:start_idx + numel].view(shape))
        start_idx += numel
    return unflattened_params

if __name__ == "__main__":

    # Initialize the model
    model = resnext50_32x4d()

    # Extract the original parameters
    original_params = [param.clone() for param in model.parameters()]
    param_shapes = [param.shape for param in original_params]

    # Flatten and then unflatten the parameters
    flat_params = flatten_parameters(original_params)
    unflattened_params = unflatten_parameters(flat_params, param_shapes)

    # Compare original and unflattened parameters
    identity_check = all(torch.allclose(orig, unflat) for orig, unflat in zip(original_params, unflattened_params))
    print(f"Identity Check: {'Passed' if identity_check else 'Failed'}")
    print(f"Number of parameters: {sum(torch.numel(param) for param in original_params)}")
