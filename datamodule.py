import albumentations as A
import random
import torch
import pytorch_lightning as pl
from torchvision import transforms
from search_view_indexible import generate_final_data
from torch.utils.data import Dataset, DataLoader


class Toy_Dataset(Dataset):
    def __init__(
        self, num_data, class_0_transform=None, class_1_transform=None
    ):
        """
        Args:
            num_data (int): Number of data samples to generate.
        """
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # Get the class index by randomly select 0 or 1
        class_id = random.randint(0, 1)
        top_view_image, search_view_indexible, class_index = generate_final_data(
            class_id=class_id
        )
        
        # make sure top_view_image is a tensor
        top_view_image = transforms.ToTensor()(top_view_image)

        return top_view_image, search_view_indexible, class_index


def custom_collate_fn(batch):
    """Custom collate function to handle different data types within a single batch.
    Args:
        batch (list): A list of tuples with (top_view_image, search_view_indexible, class_index).

    Returns:
        tuple: Contains batched images, list of indexibles, and batched class indices.
    """
    # Separate the tuple components into individual lists
    top_view_images = [item[0] for item in batch]
    search_view_indexibles = [item[1] for item in batch]
    class_indices = [item[2] for item in batch]

    # Stack the images and class indices into tensors
    top_view_images = torch.stack(top_view_images, dim=0)
    class_indices = torch.tensor(class_indices, dtype=torch.long)

    # search_view_indexibles remain as a list
    return top_view_images, search_view_indexibles, class_indices


# now write a lightning data module based on the metadata file
# and the custom collate function
class NDPI_DataModule(pl.LightningDataModule):
    def __init__(self, metadata_file, batch_size=32, num_workers=4):
        super().__init__()
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assuming you have a column 'split' in your CSV that contains 'train'/'val' labels
        if stage in (None, "fit"):
            self.train_dataset = Toy_Dataset(
                num_data=1000,
            )
            self.val_dataset = Toy_Dataset(
                num_data=100,
            )
        if stage in (None, "test"):
            self.test_dataset = Toy_Dataset(
                num_data=100,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )
