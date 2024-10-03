import openslide
import numpy as np
import torch
from toy_dataset import generate_data

class ToySearchViewIndexible:
    """ A class to represent a searchable view of a toy dataset. It is not a tensor representation of the search view or the toy dataset. 
    
    === Class Attributes ===
    --image: PIL.Image
    --downsampled_image: PIL.Image
    --downsample_rate: int
    --search_view_height: int
    --search_view_width: int
    """

    def __init__(self, image, downsampled_image, downsample_rate=16) -> None:
        self.image = image
        self.downsampled_image = downsampled_image
        self.downsample_rate = downsample_rate

        self.search_view_height, self.search_view_width = self.downsampled_image.size

    def __getitem__(self, idx):
        """Retrieve a single pixel from the image based on the (y, x) coordinates.

        Args:
        idx (tuple): A tuple of (y, x) defining the pixel coordinates.

        Returns:
        np.array: The RGB values of the extracted pixel as a numpy array.
        """
        # assert that y is in the range of the search view height and x is in the range of the search view width
        assert (
            0 <= idx[0] < self.search_view_height
        ), f"y: {idx[0]} is out of range of the search view height: {self.search_view_height}"
        assert (
            0 <= idx[1] < self.search_view_width
        ), f"x: {idx[1]} is out of range of the search view width: {self.search_view_width}"

        # Extract the pixel from the downsampled image
        pixel_values = self.downsampled_image.getpixel(idx)
        return torch.tensor(pixel_values, dtype=torch.uint8)
    
def generate_final_data(class_id):
    image, downsampled_image, class_id = generate_data(class_id)

    toy_indexible = ToySearchViewIndexible(image, downsampled_image, downsample_rate=16)

    return downsampled_image, toy_indexible, class_id

class SearchViewIndexible:
    """A class to represent a searchable view of a WSI. It is not tensor representation of the search view or the WSI.
    It is supposed to just an indexible representation of the search view that only be sparsely sampled on the CPU during training and inference.

    === Class Attributes ===
    --wsi_path: str
    --search_view_level: int
    --search_to_top_downsample_factor: int
    --search_view_height: int
    --search_view_width: int
    """

    def __init__(
        self, wsi_path, search_view_level=3, search_to_top_downsample_factor=16
    ) -> None:
        self.wsi_path = wsi_path

        self.search_view_level = search_view_level
        self.search_to_top_downsample_factor = search_to_top_downsample_factor

        self.search_view_height, self.search_view_width = openslide.OpenSlide(
            self.wsi_path
        ).level_dimensions[self.search_view_level] #TODO

    def __getitem__(self, idx):
        """Retrieve a single pixel from the slide based on the (y, x) coordinates.

        Args:
        idx (tuple): A tuple of (y, x) defining the pixel coordinates.

        Returns:
        np.array: The RGB values of the extracted pixel as a numpy array.
        """

        # assert that y is in the range of the search view height and x is in the range of the search view width
        assert (
            0 <= idx[0] < self.search_view_height
        ), f"y: {idx[0]} is out of range of the search view height: {self.search_view_height}"
        assert (
            0 <= idx[1] < self.search_view_width
        ), f"x: {idx[1]} is out of range of the search view width: {self.search_view_width}"

        try:
            slide = openslide.OpenSlide(self.wsi_path)
        except openslide.OpenSlideError as e:
            print(f"Error loading {self.wsi_path}: {e}")
            raise e

        y, x = idx
        # Extracting a region of 1x1 pixels
        region = slide.read_region(
            (
                int(x * (2**self.search_view_level)),
                int(y * (2**self.search_view_level)),
            ),
            self.search_view_level,
            (1, 1),
        )
        # Convert to numpy array, remove alpha channel, and reshape
        pixel_values = np.array(region)[:, :, :3]  # shape will be (1, 1, 3)
        pixel_values = pixel_values.reshape(3)  # reshape to (3,)
        # Convert to a torch tensor
        return torch.tensor(pixel_values, dtype=torch.uint8)

    def crop(self, TL_x, TL_y, patch_size=224):
        """Crop a patch from the search view indexible object.

        Args:
        TL_x (int): The x-coordinate of the top-left corner of the patch.
        TL_y (int): The y-coordinate of the top-left corner of the patch.
        patch_size (int): The size of the patch to crop. Defaults to 224.

        Returns:
        np.array: The RGB values of the cropped patch as a numpy array.
        """
        # assert that the top-left corner is in the range of the search view height and width
        assert (
            0 <= TL_x < self.search_view_width - patch_size
        ), f"TL_x: {TL_x} is out of range of the search view width: {self.search_view_width}"
        assert (
            0 <= TL_y < self.search_view_height - patch_size
        ), f"TL_y: {TL_y} is out of range of the search view height: {self.search_view_height}"

        try:
            slide = openslide.OpenSlide(self.wsi_path)
        except openslide.OpenSlideError as e:
            print(f"Error loading {self.wsi_path}: {e}")
            raise e

        # Extracting a region of 1x1 pixels
        region = slide.read_region(
            (
                int(TL_x * (2**self.search_view_level)),
                int(TL_y * (2**self.search_view_level)),
            ),
            self.search_view_level,
            (patch_size, patch_size),
        )

        # if RGBA, convert to RGB
        if region.mode == "RGBA":
            region = region.convert("RGB")

        # Convert to numpy array
        patch = np.array(region)

        # make sure the patch is of shape (patch_size, patch_size, 3)
        assert patch.shape == (
            patch_size,
            patch_size,
            3,
        ), f"Patch shape is {patch.shape}, expected {(patch_size, patch_size, 3)}"

        return torch.tensor(patch, dtype=torch.uint8)
