import torch


class DifferentiableIndex2DBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexable_objs, indices_batch):
        device = indices_batch.device  # Ensure device consistency

        # Ensure that indexable_objs has the same length as the batch dimension
        assert (
            len(indexable_objs) == indices_batch.shape[0]
        ), f"indexable_objs length {len(indexable_objs)} must match batch dimension {indices_batch.shape[0]}"

        # Result container
        output_batch = []
        saved_tensors = []

        # Process each item in the batch
        for i in range(len(indexable_objs)):
            indices = indices_batch[i]
            indexable_obj = indexable_objs[i]

            # Floor and ceil indices
            indices_y_floor = torch.floor(indices[:, 0]).long()
            indices_x_floor = torch.floor(indices[:, 1]).long()
            indices_y_ceil = torch.ceil(indices[:, 0]).long()
            indices_x_ceil = torch.ceil(indices[:, 1]).long()

            # Extract values at floor and ceil indices
            values_floor_floor = torch.stack(
                [
                    indexable_obj[indices_y_floor[j], indices_x_floor[j]].unsqueeze(0)
                    for j in range(len(indices))
                ],
                dim=0,
            )

            values_floor_ceil = torch.stack(
                [
                    indexable_obj[indices_y_floor[j], indices_x_ceil[j]].unsqueeze(0)
                    for j in range(len(indices))
                ],
                dim=0,
            )

            values_ceil_floor = torch.stack(
                [
                    indexable_obj[indices_y_ceil[j], indices_x_floor[j]].unsqueeze(0)
                    for j in range(len(indices))
                ],
                dim=0,
            )

            values_ceil_ceil = torch.stack(
                [
                    indexable_obj[indices_y_ceil[j], indices_x_ceil[j]].unsqueeze(0)
                    for j in range(len(indices))
                ],
                dim=0,
            )

            # Save tensors for backward pass
            saved_tensors.extend(
                [
                    indices,
                    values_floor_floor,
                    values_floor_ceil,
                    values_ceil_floor,
                    values_ceil_ceil,
                ]
            )

            # Bilinear interpolation # TODO okay this may not be actually correct, please fix it.
            weights_y_floor = indices[:, 0] - indices_y_floor.float().to(device)
            weights_y_ceil = indices_y_ceil.float().to(device) - indices[:, 0]
            weights_x_floor = indices[:, 1] - indices_x_floor.float().to(device)
            weights_x_ceil = indices_x_ceil.float().to(device) - indices[:, 1]

            weights_x_ceil = weights_x_ceil.unsqueeze(1).unsqueeze(
                2
            )  # Shape: [Nk, 1, 1]
            weights_x_floor = weights_x_floor.unsqueeze(1).unsqueeze(
                2
            )  # Shape: [Nk, 1, 1]

            # do the same for the rest of the weights
            weights_y_ceil = weights_y_ceil.unsqueeze(1).unsqueeze(
                2
            )  # Shape: [Nk, 1, 1]
            weights_y_floor = weights_y_floor.unsqueeze(1).unsqueeze(
                2
            )  # Shape: [Nk, 1, 1]

            # move values_floor_floor to the same device as weights_x_ceil
            values_floor_floor = values_floor_floor.to(weights_x_ceil.device)
            # do the same for the rest of the values
            values_floor_ceil = values_floor_ceil.to(weights_x_ceil.device)
            values_ceil_floor = values_ceil_floor.to(weights_x_ceil.device)
            values_ceil_ceil = values_ceil_ceil.to(weights_x_ceil.device)

            # the shape of the weights_x_ceil is torch.Size([Nk]) and the shape of the values_floor_floor is torch.Size([Nk, 1, 3])
            # TODO make sure you check the correctness of the linear interpolation here
            interpolated_y_floor = (
                weights_x_ceil * values_floor_floor
                + weights_x_floor * values_floor_ceil
            )

            interpolated_y_ceil = (
                weights_x_ceil * values_ceil_floor + weights_x_floor * values_ceil_ceil
            )

            output = (
                weights_y_ceil * interpolated_y_floor
                + weights_y_floor * interpolated_y_ceil
            )
            output_batch.append(output)

        # Save all necessary tensors for the backward pass
        ctx.save_for_backward(*saved_tensors)

        # current output has shape [b, Nk, 1, 3], we need to make it torch.Size([b, Nk, 3])
        for i in range(len(output_batch)):
            output_batch[i] = output_batch[i].squeeze(1)
            # print(output_batch[i].shape)

        output = torch.stack(output_batch, dim=0)

        # Stack output for the batch
        return output

    @staticmethod
    def backward(ctx, grad_output_batch):
        saved_tensors = ctx.saved_tensors

        # print(f"shape of grad_output_batch: {grad_output_batch.shape}")

        # Gradient container for indices_batch
        grad_indices_batch = []

        # Each item in the batch has 5 saved tensors
        num_saved_tensors_per_item = 5

        batch_size = grad_output_batch.shape[0]

        for i in range(batch_size):
            start_idx = i * num_saved_tensors_per_item
            end_idx = start_idx + num_saved_tensors_per_item

            (
                indices,
                values_floor_floor,
                values_floor_ceil,
                values_ceil_floor,
                values_ceil_ceil,
            ) = saved_tensors[start_idx:end_idx]

            grad_output = grad_output_batch[i]

            # Ensure that tensors are on the correct device
            values_floor_floor = values_floor_floor.to(grad_output.device)
            values_floor_ceil = values_floor_ceil.to(grad_output.device)
            values_ceil_floor = values_ceil_floor.to(grad_output.device)
            values_ceil_ceil = values_ceil_ceil.to(
                grad_output.device
            )  # TODO check that this is indeed correct

            # print(f"Values floor floor shape: {values_floor_floor.shape}")
            # print(f"Values floor ceil shape: {values_floor_ceil.shape}")
            # print(f"Values ceil floor shape: {values_ceil_floor.shape}")
            # print(f"Values ceil ceil shape: {values_ceil_ceil.shape}")

            # Calculate gradients for indices
            grad_indices_y_mat = (values_ceil_floor + values_ceil_ceil) - (
                values_floor_floor + values_floor_ceil
            )
            grad_indices_x_mat = (values_floor_ceil + values_ceil_ceil) - (
                values_floor_floor + values_ceil_floor
            )

            # the current shape is [Nk, 1, 3], we need to make it torch.Size([Nk, 3])
            grad_indices_y_mat = grad_indices_y_mat.squeeze(1)
            grad_indices_x_mat = grad_indices_x_mat.squeeze(1)

            # print(f"Shape of grad_indices_y_mat: {grad_indices_y_mat.shape}")
            # print(f"Shape of grad_indices_x_mat: {grad_indices_x_mat.shape}")

            # stack the gradients for y and x along the dim 1
            grad_indices_mat = torch.stack(
                [grad_indices_y_mat, grad_indices_x_mat], dim=1
            )

            # print(f"Shape of grad_indices_mat: {grad_indices_mat.shape}")
            # print(f"Shape of grad_output: {grad_output.shape}")
            # print(f"Shape of indices: {grad_indices_mat.shape}")

            # Ensure indices_mat is a float tensor
            if grad_indices_mat.dtype != torch.float32:
                grad_indices_mat = grad_indices_mat.float()

            # Ensure grad_output is a float tensor
            if grad_output.dtype != torch.float32:
                grad_output = grad_output.float()

            # grad_output has shape [Nk, 3], indices_mat has shape [Nk, 2, 3]
            grad_indices = torch.bmm(
                grad_indices_mat, grad_output.unsqueeze(-1)
            )  # Shape: [Nk, 2, 1]

            # Remove the last dimension to get the final gradient shape [Nk, 2]
            grad_indices = grad_indices.squeeze(-1)  # Shape: [Nk, 2]

            # print(f"Shape of grad_indices: {grad_indices.shape}")

            grad_indices_batch.append(grad_indices)

        grad_indices_stacked = torch.stack(grad_indices_batch, dim=0)

        # print(f"Stacked grad_indices shape: {grad_indices_stacked.shape}")

        # No gradient for indexable_objs
        grad_indexable_objs = None

        # Stack gradients for the batch
        return (
            grad_indexable_objs,
            grad_indices_stacked,
        )


# Example use in your model
def differentiable_index_2d_batch(indexable_objs, indices_batch):
    return DifferentiableIndex2DBatchFunction.apply(indexable_objs, indices_batch)


class DifferentiableCrop2DBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexable_objs, indices_batch, patch_size=224):
        device = indices_batch.device  # Ensure device consistency

        # Ensure that indexable_objs has the same length as the batch dimension
        assert (
            len(indexable_objs) == indices_batch.shape[0]
        ), f"indexable_objs length {len(indexable_objs)} must match batch dimension {indices_batch.shape[0]}"

        # Result container
        output_batch = []
        saved_tensors = []

        # Process each item in the batch
        for i in range(len(indexable_objs)):
            indices = indices_batch[i]
            indexable_obj = indexable_objs[i]

            central_indices_y_floor = torch.floor(indices[:, 0]).long()
            central_indices_x_floor = torch.floor(indices[:, 1]).long()
            central_indices_y_ceil = torch.ceil(indices[:, 0]).long()
            central_indices_x_ceil = torch.ceil(indices[:, 1]).long()

            # calculate the TL_indices floor and ceil by subtracting half of the patch size
            TL_indices_y_floor = central_indices_y_floor - patch_size // 2
            TL_indices_x_floor = central_indices_x_floor - patch_size // 2
            TL_indices_y_ceil = central_indices_y_ceil - patch_size // 2
            TL_indices_x_ceil = central_indices_x_ceil - patch_size // 2

            # extract the patch_size x patch_size patches
            # use indexable_obj.crop(self, TL_x, TL_y, patch_size=224) method, which returns a tensor of shape [patch_size, patch_size, 3]
            # the shape of patches_floor_floor is (len(indices), patch_size, patch_size, 3)
            patches_floor_floor = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_floor[j],
                        TL_indices_y_floor[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            assert patches_floor_floor.shape == (
                len(indices),
                patch_size,
                patch_size,
                3,
            )

            patches_floor_ceil = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_ceil[j],
                        TL_indices_y_floor[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            patches_ceil_floor = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_floor[j],
                        TL_indices_y_ceil[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            patches_ceil_ceil = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_ceil[j],
                        TL_indices_y_ceil[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            # Save tensors for backward pass
            saved_tensors.extend(
                [
                    indices,
                    patches_floor_floor,
                    patches_floor_ceil,
                    patches_ceil_floor,
                    patches_ceil_ceil,
                ]
            )

            # Bilinear interpolation
            weights_y_floor = indices[:, 0] - TL_indices_y_floor.float().to(device)
            weights_y_ceil = TL_indices_y_ceil.float().to(device) - indices[:, 0]
            weights_x_floor = indices[:, 1] - TL_indices_x_floor.float().to(device)
            weights_x_ceil = TL_indices_x_ceil.float().to(device) - indices[:, 1]

            weights_x_ceil = weights_x_ceil.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]
            weights_x_floor = weights_x_floor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]
            weights_y_ceil = weights_y_ceil.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]
            weights_y_floor = weights_y_floor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]

            # move patches_floor_floor to the same device as weights_x_ceil
            patches_floor_floor = patches_floor_floor.to(weights_x_ceil.device)
            # do the same for the rest of the patches
            patches_floor_ceil = patches_floor_ceil.to(weights_x_ceil.device)
            patches_ceil_floor = patches_ceil_floor.to(weights_x_ceil.device)
            patches_ceil_ceil = patches_ceil_ceil.to(weights_x_ceil.device)

            # print(f"Shape of patches_floor_floor: {patches_floor_floor.shape}")
            # print(f"Shape of weights_x_ceil: {weights_x_ceil.shape}")

            # the shape of the weights_x_ceil is torch.Size([len(indices)]) and the shape of the patches_floor_floor is torch.Size([len(indices), patch_size, patch_size, 3])
            interpolated_y_floor = (
                weights_x_ceil * patches_floor_floor
                + weights_x_floor * patches_floor_ceil
            )

            # print(f"Shape of interpolated_y_floor: {interpolated_y_floor.shape}")

            interpolated_y_ceil = (
                weights_x_ceil * patches_ceil_floor
                + weights_x_floor * patches_ceil_ceil
            )

            output = (
                weights_y_ceil * interpolated_y_floor
                + weights_y_floor * interpolated_y_ceil
            )
            output_batch.append(output)

        # Save all necessary tensors for the backward pass
        ctx.save_for_backward(*saved_tensors)

        output = torch.stack(output_batch, dim=0)
        assert output.shape == (
            len(indexable_objs),
            len(indices_batch[0]),
            patch_size,
            patch_size,
            3,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved_tensors = ctx.saved_tensors

        # Gradient container for indices_batch
        grad_indices_batch = []

        # Each item in the batch has 5 saved tensors
        num_saved_tensors_per_item = 5

        batch_size = grad_output.shape[0]

        for i in range(batch_size):
            start_idx = i * num_saved_tensors_per_item
            end_idx = (
                start_idx + num_saved_tensors_per_item
            )  # this is just to calculate the actual position of the saved_tensors in the big list

            (
                indices,
                patches_floor_floor,
                patches_floor_ceil,
                patches_ceil_floor,
                patches_ceil_ceil,
            ) = saved_tensors[start_idx:end_idx]

            grad_output_item = grad_output[i]

            # Ensure that tensors are on the correct device
            patches_floor_floor = patches_floor_floor.to(grad_output_item.device)
            patches_floor_ceil = patches_floor_ceil.to(grad_output_item.device)
            patches_ceil_floor = patches_ceil_floor.to(grad_output_item.device)
            patches_ceil_ceil = patches_ceil_ceil.to(grad_output_item.device)

            # Calculate gradients for indices
            grad_indices_y_mat = (patches_ceil_floor + patches_ceil_ceil) - (
                patches_floor_floor + patches_floor_ceil
            )

            grad_indices_x_mat = (patches_floor_ceil + patches_ceil_ceil) - (
                patches_floor_floor + patches_ceil_floor
            )

            # stack the gradients for y and x along the dim 1
            grad_indices_mat = torch.stack(
                [grad_indices_y_mat, grad_indices_x_mat], dim=1
            )

            # Ensure indices_mat is a float tensor
            if grad_indices_mat.dtype != torch.float32:
                grad_indices_mat = grad_indices_mat.float()

            # Ensure grad_output is a float tensor
            if grad_output_item.dtype != torch.float32:
                grad_output_item = grad_output_item.float()

            # print("grad_indices_mat shape: ", grad_indices_mat.shape)
            # print("grad_output_item shape: ", grad_output_item.shape)

            # Flatten the spatial dimensions (224 * 224 * 3 = 150528)
            grad_indices_mat_flat = grad_indices_mat.view(
                8, 2, -1
            )  # Shape: [8, 2, 150528] 150528 or would be whatever flatten dimension of the patch image is
            grad_output_item_flat = grad_output_item.view(8, -1)  # Shape: [8, 150528]

            # Perform batch matrix multiplication
            output = torch.bmm(
                grad_indices_mat_flat, grad_output_item_flat.unsqueeze(2)
            )  # Shape: [8, 2, 1]

            # Remove the last dimension to get the desired shape [8, 2]
            output = output.squeeze(2)

            grad_indices_batch.append(output)

        grad_indices_stacked = torch.stack(grad_indices_batch, dim=0)

        # No gradient for indexable_objs
        grad_indexable_objs = None

        # Stack gradients for the batch
        return (
            grad_indexable_objs,
            grad_indices_stacked,
            None,
        )


def differentiable_crop_2d_batch(indexable_objs, indices_batch, patch_size=224):
    return DifferentiableCrop2DBatchFunction.apply(
        indexable_objs, indices_batch, patch_size
    )
