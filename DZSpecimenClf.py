import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
from differentiable_indexing import differentiable_crop_2d_batch
from PIL import ImageDraw
from torchvision import transforms


class Attn(nn.Module):
    def __init__(self, head_dim, use_flash_attention):
        super(Attn, self).__init__()
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention

    def forward(self, q, k, v):
        if self.use_flash_attention:
            # Use PyTorch's built-in scaled dot product attention with flash attention support
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Compute scaled dot product attention manually
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        return attn_output


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_heads=8,
        use_flash_attention=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)

        # The attention mechanism
        self.attn = Attn(
            head_dim=self.head_dim, use_flash_attention=use_flash_attention
        )

        # Class token for classification
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        # Shape of x: (batch_size, N, d_model), where N is the sequence length

        batch_size = x.size(0)

        # Prepare the class token (batch_size, 1, d_model)
        class_token = self.class_token.expand(batch_size, -1, -1)

        # Concatenate the class token with the input tokens (batch_size, N+1, d_model)
        x = torch.cat((class_token, x), dim=1)

        # Linear projections for Q, K, V (batch_size, num_heads, N+1, head_dim)
        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply attention (batch_size, num_heads, N+1, head_dim)
        attn_output = self.attn(q, k, v)

        # Concatenate attention output across heads (batch_size, N+1, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Apply final linear projection
        x = self.out_proj(attn_output)

        # Extract the class token's output (batch_size, d_model)
        class_token_output = x[:, 0, :]

        return class_token_output


# def get_square_coordinates(center_tensor, square_side_length=224):
#     """center_tensor is of shape [b, N, 2], where the last dimension is the center of the square.
#     Create a tensor of shape [b, N square_side_length**2, 2] where the last dimension is the coordinates of the square centered at the center_tensor with side length square_side_length.
#     The coordinates are like center - 1, center - 2, etc. to center + 1, center + 2, etc.

#     preconditions:
#     - square_side_length is an even number
#     """

#     # assert that square_side_length is an even number
#     assert (
#         square_side_length % 2 == 0
#     ), f"square_side_length should be an even number, but is {square_side_length}"

#     # get the batch size and the number of centers
#     b, N, _ = center_tensor.shape

#     # create a tensor of shape [square_side_length, square_side_length] with the coordinates of the square
#     square_coordinates = torch.stack(
#         torch.meshgrid(
#             torch.arange(-square_side_length // 2, square_side_length // 2),
#             torch.arange(-square_side_length // 2, square_side_length // 2),
#         ),
#         dim=-1,
#     )

#     # reshape the square_coordinates to have the shape [1, square_side_length**2, 2]
#     square_coordinates = square_coordinates.view(1, -1, 2)

#     # create a tensor of shape [b, N, square_side_length**2, 2] with the coordinates of the square
#     square_coordinates = square_coordinates.repeat(b, N, 1, 1)

#     # add the center_tensor to the square_coordinates
#     # move the square_coordinates to the same device as the center_tensor
#     square_coordinates = square_coordinates.to(center_tensor.device)
#     square_coordinates = center_tensor.unsqueeze(2) + square_coordinates

#     return square_coordinates


class DZSpecimenClf(nn.Module):
    def __init__(self, N, patch_size=224, num_classes=2):
        super(DZSpecimenClf, self).__init__()
        # Load the pretrained ResNeXt50 model
        self.resnext50 = models.resnext50_32x4d(pretrained=False)
        self.resnext50.fc = nn.Linear(self.resnext50.fc.in_features, N * 2)

        self.patch_feature_extractor = models.resnext50_32x4d(pretrained=False)
        # do not need the last layer of linear forward
        self.patch_feature_extractor.fc = nn.Identity()
        self.aggregator = MultiHeadAttentionClassifier(d_model=2048, num_heads=8)

        self.last_layer = nn.Linear(2048, num_classes)
        self.num_classes = num_classes

        # Define a sigmoid activation layer
        self.sigmoid = nn.Sigmoid()

        # initialize a trainable tensor of shape (N, k, 1)
        self.N = N
        self.patch_size = patch_size

    def forward(self, topview_image_tensor, search_view_indexibles):
        # Pass input through the feature extractor part
        x = self.resnext50(topview_image_tensor)  # x should have shape [b, N*2]

        x = x.view(
            x.size(0), -1, 2
        )  # now after reshaping, x should have shape [b, N, 2]

        # assert that the output is of the correct shape
        assert (
            x.shape[1] == self.N and x.shape[2] == 2
        ), f"Output shape is {x.shape}, rather than the expected ({self.N}, 2)"

        # apply the sigmoid activation
        x = self.sigmoid(x)

        search_view_heights = [
            search_view_indexible.search_view_height - 1
            for search_view_indexible in search_view_indexibles
        ]
        search_view_widths = [
            search_view_indexible.search_view_width - 1
            for search_view_indexible in search_view_indexibles
        ]

        # padded_search_view_heights will be search_view_heights subtracted by patch_size
        padded_search_view_heights = [
            search_view_height - self.patch_size
            for search_view_height in search_view_heights
        ]
        # padded_search_view_widths will be search_view_widths subtracted by patch_size
        padded_search_view_widths = [
            search_view_width - self.patch_size
            for search_view_width in search_view_widths
        ]

        assert (
            len(search_view_heights)
            == len(search_view_widths)
            == len(search_view_indexibles)
            == x.shape[0]
        ), f"Batch dim / length of search_view_heights: {len(search_view_heights)}, search_view_widths: {len(search_view_widths)}, search_view_indexibles: {len(search_view_indexibles)}, x: {x.shape[0]}"

        search_view_heights_tensor = (
            torch.tensor(padded_search_view_heights).view(-1, 1, 1).to(x.device)
        )
        search_view_widths_tensor = (
            torch.tensor(padded_search_view_widths).view(-1, 1, 1).to(x.device)
        )
        # x is a bunch of y, x coordinates there are b, N*k of them, multiply y by the search view height and x by the search view width
        # Scale x by multiplying the y and x coordinates by the respective dimensions
        # First column of x are y coordinates, second column are x coordinates

        x_scaled = (x[..., 0].unsqueeze(-1) * search_view_heights_tensor).squeeze(-1)
        y_scaled = (x[..., 1].unsqueeze(-1) * search_view_widths_tensor).squeeze(-1)

        # now add patch_size // 2 to the x_scaled and y_scaled tensors
        x_scaled = x_scaled + self.patch_size // 2
        y_scaled = y_scaled + self.patch_size // 2

        # now stack the x_scaled and y_scaled tensors along the last dimension
        xy = torch.stack([x_scaled, y_scaled], dim=-1)

        # Continue with x_scaled instead of x
        x = differentiable_crop_2d_batch(search_view_indexibles, xy)

        # assert that x has shape [b, N, patch_size, patch_size, 3]
        assert (
            x.shape[1] == self.N
            and x.shape[2] == self.patch_size
            and x.shape[3] == self.patch_size
            and x.shape[4] == 3
        ), f"Output shape is {x.shape}, rather than the expected (b, N, {self.patch_size}, {self.patch_size}, 3)"

        # now reshape the indexing_output to have the shape (b*N, patch_size, patch_size, 3)
        x = x.view(x.shape[0] * x.shape[1], self.patch_size, self.patch_size, 3)

        # now reshape the indexing_output to have the shape (b*N, 3, patch_size, patch_size)
        x = x.permute(0, 3, 1, 2)

        # apply the resnext50 feature extractor to the patch
        x = self.patch_feature_extractor(x)

        # reshape the output to shape (b*N, 2048)
        x = x.view(x.shape[0], -1)

        # reshape the output to shape (b, N, 2048)
        x = x.view(x.shape[0] // self.N, self.N, -1)

        # apply a flash transformer to the N tokens of token size 2048 and keep the class token
        x = self.aggregator(x)
        # assert that x has shape [b, 2048]
        assert (
            x.shape[1] == 2048
        ), f"Output shape is {x.shape}, rather than the expected (b, 2048)"

        # apply the last layer to the output
        x = self.last_layer(x)

        return x

    def get_patches(self, topview_image_tensor, search_view_indexibles):
        # Pass input through the feature extractor part
        x = self.resnext50(topview_image_tensor)  # x should have shape [b, N*2]

        x = x.view(
            x.size(0), -1, 2
        )  # now after reshaping, x should have shape [b, N, 2]

        # assert that the output is of the correct shape
        assert (
            x.shape[1] == self.N and x.shape[2] == 2
        ), f"Output shape is {x.shape}, rather than the expected ({self.N}, 2)"

        # apply the sigmoid activation
        x = self.sigmoid(x)

        search_view_heights = [
            search_view_indexible.search_view_height - 1
            for search_view_indexible in search_view_indexibles
        ]
        search_view_widths = [
            search_view_indexible.search_view_width - 1
            for search_view_indexible in search_view_indexibles
        ]

        # padded_search_view_heights will be search_view_heights subtracted by patch_size
        padded_search_view_heights = [
            search_view_height - self.patch_size
            for search_view_height in search_view_heights
        ]
        # padded_search_view_widths will be search_view_widths subtracted by patch_size
        padded_search_view_widths = [
            search_view_width - self.patch_size
            for search_view_width in search_view_widths
        ]

        assert (
            len(search_view_heights)
            == len(search_view_widths)
            == len(search_view_indexibles)
            == x.shape[0]
        ), f"Batch dim / length of search_view_heights: {len(search_view_heights)}, search_view_widths: {len(search_view_widths)}, search_view_indexibles: {len(search_view_indexibles)}, x: {x.shape[0]}"

        search_view_heights_tensor = (
            torch.tensor(padded_search_view_heights).view(-1, 1, 1).to(x.device)
        )
        search_view_widths_tensor = (
            torch.tensor(padded_search_view_widths).view(-1, 1, 1).to(x.device)
        )
        # x is a bunch of y, x coordinates there are b, N*k of them, multiply y by the search view height and x by the search view width
        # Scale x by multiplying the y and x coordinates by the respective dimensions
        # First column of x are y coordinates, second column are x coordinates

        x_scaled = (x[..., 0].unsqueeze(-1) * search_view_heights_tensor).squeeze(-1)
        y_scaled = (x[..., 1].unsqueeze(-1) * search_view_widths_tensor).squeeze(-1)

        # now add patch_size // 2 to the x_scaled and y_scaled tensors
        x_scaled = x_scaled + self.patch_size // 2
        y_scaled = y_scaled + self.patch_size // 2

        # now stack the x_scaled and y_scaled tensors along the last dimension
        xy = torch.stack([x_scaled, y_scaled], dim=-1)

        # Continue with x_scaled instead of x
        x = differentiable_crop_2d_batch(search_view_indexibles, xy)

        # assert that x has shape [b, N, patch_size, patch_size, 3]
        assert (
            x.shape[1] == self.N
            and x.shape[2] == self.patch_size
            and x.shape[3] == self.patch_size
            and x.shape[4] == 3
        ), f"Output shape is {x.shape}, rather than the expected (b, N, {self.patch_size}, {self.patch_size}, 3)"

        # convert to a list of list of PIL images
        x = x.cpu().numpy()
        x = x.tolist()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = transforms.ToPILImage()(x[i, j])

        return x

    # def sampling_points(self, topview_image_tensor, search_view_indexibles):
    #     # Pass input through the feature extractor part
    #     x = self.resnext50(topview_image_tensor)

    #     x = x.view(x.size(0), -1, 2)

    #     # assert that the output is of the correct shape
    #     assert (
    #         x.shape[1] == self.N * self.k and x.shape[2] == 2
    #     ), f"Output shape is {x.shape}, rather than the expected ({self.N * self.k}, 2)"

    #     # apply the sigmoid activation
    #     x = self.sigmoid(x)

    #     search_view_heights = [
    #         search_view_indexible.search_view_height - 1
    #         for search_view_indexible in search_view_indexibles
    #     ]

    #     search_view_widths = [
    #         search_view_indexible.search_view_width - 1
    #         for search_view_indexible in search_view_indexibles
    #     ]

    #     assert (
    #         len(search_view_heights)
    #         == len(search_view_widths)
    #         == len(search_view_indexibles)
    #         == x.shape[0]
    #     ), f"Batch dim / length of search_view_heights: {len(search_view_heights)}, search_view_widths: {len(search_view_widths)}, search_view_indexibles: {len(search_view_indexibles)}, x: {x.shape[0]}"

    #     search_view_heights_tensor = (
    #         torch.tensor(search_view_heights).view(-1, 1, 1).to(x.device)
    #     )
    #     search_view_widths_tensor = (
    #         torch.tensor(search_view_widths).view(-1, 1, 1).to(x.device)
    #     )

    #     # x is a bunch of y, x coordinates there are b, N*k of them, multiply y by the search view height and x by the search view width
    #     # Scale x by multiplying the y and x coordinates by the respective dimensions
    #     # First column of x are y coordinates, second column are x coordinates

    #     x_scaled = (x[..., 0].unsqueeze(-1) * search_view_heights_tensor).squeeze(-1)
    #     y_scaled = (x[..., 1].unsqueeze(-1) * search_view_widths_tensor).squeeze(-1)

    #     # now stack the x_scaled and y_scaled tensors along the last dimension
    #     xy = torch.stack([x_scaled, y_scaled], dim=-1)

    #     return xy

    # def visualize_sampling_points(self, topview_image_tensor, search_view_indexibles):
    #     xy = self.sampling_points(topview_image_tensor, search_view_indexibles)

    #     # annotate the points xy as red dots on the each images in the topview_image_tensor
    #     # return a list of PIL images with the annotated points

    #     # need to downsize the coordinates by a factor of 2**4

    #     # downsample the coordinates by a factor of 2**4
    #     xy = xy / 2**4

    #     # convert the tensor to a list of numpy arrays
    #     xy = xy.cpu().numpy()

    #     # convert the tensor to a list of PIL images
    #     topview_images = [
    #         transforms.ToPILImage()(topview_image)
    #         for topview_image in topview_image_tensor
    #     ]

    #     # draw the points on the images
    #     annotated_images = []

    #     for topview_image, xy_points in zip(topview_images, xy):
    #         draw = ImageDraw.Draw(topview_image)
    #         for xy_point in xy_points:
    #             draw.ellipse(
    #                 (
    #                     xy_point[1] - 2,
    #                     xy_point[0] - 2,
    #                     xy_point[1] + 2,
    #                     xy_point[0] + 2,
    #                 ),
    #                 fill="red",
    #             )
    #         annotated_images.append(topview_image)

    #     return annotated_images
