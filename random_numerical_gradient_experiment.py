import torch
import numpy as np
import csv
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import resnext50_32x4d
from param_flat import flatten_parameters, unflatten_parameters
from scipy.optimize import approx_fprime


def compute_numerical_gradient(model, input_data, target_data, loss_fn, epsilon=1e-5, n_params=None):
    # Store numerical gradients for all parameters
    numerical_gradients = []

    # Get model parameters and their shapes
    params = list(model.parameters())
    param_shapes = [param.shape for param in params]
    total_params = sum(param.numel() for param in params)

    # Randomly select a subset of parameter indices for gradient computation
    if n_params:
        param_indices = np.random.choice(total_params, size=n_params, replace=False)
    else:
        param_indices = np.arange(total_params)

    # Flatten all parameters into a single tensor
    param_list = flatten_parameters(params)
    
    # Compute the numerical gradients using epsilon perturbations
    for idx in tqdm(param_indices, desc="Computing numerical gradients"):
        # Save the original parameter value
        original_value = param_list[idx].item()

        # Perturb parameter positively and compute the loss
        param_list[idx] = original_value + epsilon
        set_model_params(model, param_list, param_shapes)
        output_plus = model(input_data)
        loss_plus = loss_fn(output_plus, target_data).item()

        # Perturb parameter negatively and compute the loss
        param_list[idx] = original_value - epsilon
        set_model_params(model, param_list, param_shapes)
        output_minus = model(input_data)
        loss_minus = loss_fn(output_minus, target_data).item()

        # Restore the original parameter value
        param_list[idx] = original_value
        set_model_params(model, param_list, param_shapes)

        # Compute the numerical gradient
        grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
        numerical_gradients.append(grad_approx)

    return numerical_gradients, param_indices


def compute_approx_fprime_gradient(model, input_data, target_data, loss_fn, epsilon=1e-5):
    """Compute numerical gradient using scipy's approx_fprime."""
    params = list(model.parameters())
    param_shapes = [param.shape for param in params]
    flat_params = flatten_parameters(params).detach().cpu().numpy()

    def loss_wrapper(flat_params):
        """Wrapper function that sets the model parameters and returns the loss."""
        set_model_params(model, torch.tensor(flat_params, dtype=torch.float32), param_shapes)
        output = model(input_data)
        return loss_fn(output, target_data).item()

    # Compute gradients using approx_fprime
    grad_approximations = approx_fprime(flat_params, loss_wrapper, epsilon)
    return grad_approximations


def set_model_params(model, flat_params, param_shapes):
    """Set the model parameters using a flattened parameter tensor."""
    unflattened_params = unflatten_parameters(flat_params, param_shapes)
    with torch.no_grad():
        for param, unflattened_param in zip(model.parameters(), unflattened_params):
            param.copy_(unflattened_param)


def compute_backward_gradient(model, input_data, target_data, loss_fn):
    output = model(input_data)
    loss = loss_fn(output, target_data)
    model.zero_grad()
    loss.backward()
    backward_gradients = [param.grad.clone() for param in model.parameters()]
    return backward_gradients


def compare_gradients(
    numerical_gradients, approx_fprime_gradients, backward_gradients, param_indices, csv_filename="grad_comparison.csv"
):
    device = "cpu"
    numerical_gradients = [grad for grad in numerical_gradients]
    approx_fprime_gradients = [grad for grad in approx_fprime_gradients]
    backward_gradients = [grad.to(device) for grad in backward_gradients]

    # Flatten all gradients for comparison
    backward_grad_flat = torch.cat([grad.flatten() for grad in backward_gradients]).cpu().numpy()

    # Compare only the selected parameter gradients
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter Index", "Numerical Gradient", "Approx Fprime Gradient", "Backward Gradient", "Relative Error (Numerical)", "Relative Error (Approx Fprime)"])

        for i, idx in enumerate(param_indices):
            num_grad = numerical_gradients[i]
            approx_fprime_grad = approx_fprime_gradients[idx]
            back_grad = backward_grad_flat[idx]

            rel_error_numerical = np.linalg.norm(back_grad - num_grad) / (
                np.linalg.norm(back_grad) + np.linalg.norm(num_grad) + 1e-8
            )
            rel_error_approx = np.linalg.norm(back_grad - approx_fprime_grad) / (
                np.linalg.norm(back_grad) + np.linalg.norm(approx_fprime_grad) + 1e-8
            )

            writer.writerow([idx, num_grad, approx_fprime_grad, back_grad, rel_error_numerical, rel_error_approx])

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

    print("Computing approx fprime gradient...")
    approx_fprime_gradients = compute_approx_fprime_gradient(
        model, input_data, target_data, loss_fn, epsilon=1e-5
    )

    print("Computing backward gradient...")
    backward_gradients = compute_backward_gradient(model, input_data, target_data, loss_fn)

    print("Comparing gradients...")
    compare_gradients(numerical_gradients, approx_fprime_gradients, backward_gradients, param_indices)
