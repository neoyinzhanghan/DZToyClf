import torch
import numpy as np
import random
import csv
import torch.nn as nn
from torchvision.models import resnext50_32x4d


def compute_numerical_gradient(
    model, input_data, target_data, loss_fn, epsilon=1e-5, n_params=None
):
    # Store numerical gradients for all parameters
    numerical_gradients = []

    # Get model parameters
    params = list(model.parameters())

    # Flatten all parameters into a 1D array and count total parameters
    total_params = sum(param.numel() for param in params)

    # Randomly select a subset of parameter indices for gradient computation
    if n_params:
        param_indices = np.random.choice(total_params, size=n_params, replace=False)
    else:
        param_indices = np.arange(total_params)

    grads = []

    for param_index in param_indices:
        # clone the model parameters
        params_clone_plus = [param.clone() for param in params]
        params_clone_minus = [param.clone() for param in params]

        # flatten the cloned parameters
        params_flat_plus = np.concatenate(
            [param.flatten().cpu().detach().numpy() for param in params_clone_plus]
        )
        params_flat_minus = np.concatenate(
            [param.flatten().cpu().detach().numpy() for param in params_clone_minus]
        )

        # perturb the selected parameter
        params_flat_plus[param_index] += epsilon
        params_flat_minus[param_index] -= epsilon

        # unflatten the parameters
        start_idx = 0
        for param, param_clone_plus, param_clone_minus in zip(
            params, params_clone_plus, params_clone_minus
        ):
            numel = param.numel()
            param_clone_plus.data = torch.tensor(
                params_flat_plus[start_idx : start_idx + numel].reshape(param.shape)
            )
            param_clone_minus.data = torch.tensor(
                params_flat_minus[start_idx : start_idx + numel].reshape(param.shape)
            )
            start_idx += numel

        # Perform forward pass to compute the output and loss
        with torch.no_grad():
            for original_param, clone_param in zip(
                model.parameters(), params_clone_plus
            ):
                original_param.copy_(clone_param)
        output_plus = model(input_data)

        with torch.no_grad():
            for original_param, clone_param in zip(
                model.parameters(), params_clone_minus
            ):
                original_param.copy_(clone_param)
        output_minus = model(input_data)

        with torch.no_grad():
            for original_param, saved_param in zip(model.parameters(), params):
                original_param.copy_(saved_param)
        loss_plus = loss_fn(output_plus, target_data)
        loss_minus = loss_fn(output_minus, target_data)

        # Compute numerical gradient
        grad_approx = (loss_plus.item() - loss_minus.item()) / (2 * epsilon)
        grads.append(grad_approx)

    return grads, param_indices


def compute_backward_gradient(model, input_data, target_data, loss_fn):
    output = model(input_data)
    loss = loss_fn(output, target_data)
    model.zero_grad()
    loss.backward()
    backward_gradients = [param.grad.clone() for param in model.parameters()]
    return backward_gradients


def compare_gradients(
    numerical_gradients, backward_gradients, param_indices, csv_filename="grad_comparison.csv"
):
    device = "cpu"
    numerical_gradients = [grad for grad in numerical_gradients]
    backward_gradients = [grad.to(device) for grad in backward_gradients]

    # Flatten all gradients for comparison
    numerical_grad_flat = np.array(numerical_gradients)
    backward_grad_flat = np.concatenate(
        [grad.flatten().cpu().numpy() for grad in backward_gradients]
    )

    # Compare only the selected parameter gradients
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter Index", "Numerical Gradient", "Backward Gradient", "Relative Error"])

        for i, idx in enumerate(param_indices):
            num_grad = numerical_grad_flat[i]
            back_grad = backward_grad_flat[idx]
            relative_error = np.linalg.norm(back_grad - num_grad) / (
                np.linalg.norm(back_grad) + np.linalg.norm(num_grad) + 1e-8
            )
            writer.writerow([idx, num_grad, back_grad, relative_error])

    print(f"Gradient comparison saved to {csv_filename}")


if __name__ == "__main__":
    # Using a ResNeXt-50 model from torchvision
    model = resnext50_32x4d(num_classes=2)
    loss_fn = nn.CrossEntropyLoss()

    # Generate random input and target data
    input_data = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    target_data = torch.tensor([1])  # Example target class

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)
    target_data = target_data.to(device)

    print("Computing numerical gradient...")

    # Move everything to CPU for numerical gradient calculation
    model.to("cpu")
    input_data = input_data.to("cpu")
    target_data = target_data.to("cpu")

    # Compute numerical gradients for a subset of parameters
    N_params = 100  # Number of randomly selected parameters for numerical gradient calculation
    numerical_gradients, param_indices = compute_numerical_gradient(
        model, input_data, target_data, loss_fn, n_params=N_params
    )

    print("Computing backward gradient...")
    backward_gradients = compute_backward_gradient(model, input_data, target_data, loss_fn)

    print("Comparing gradients...")
    compare_gradients(numerical_gradients, backward_gradients, param_indices)
