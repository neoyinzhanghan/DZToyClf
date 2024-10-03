import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, AUROC
import pytorch_lightning as pl
from torchvision.models import resnext50_32x4d
from torch.utils.data import DataLoader
from datamodule import NDPI_DataModule  # Assuming you saved the DataModule from the previous code

class ResNeXtLightningModule(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-4, T_max=10):
        super(ResNeXtLightningModule, self).__init__()
        self.save_hyperparameters()

        # Initialize ResNeXt-50
        self.model = resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.f1_score = F1Score(num_classes=num_classes, task="multiclass")
        self.auroc = AUROC(num_classes=num_classes, task="multiclass")

    def forward(self, topview_image_tensor):
        return self.model(topview_image_tensor)

    def training_step(self, batch, batch_idx):
        topview_image, _, class_index = batch
        outputs = self(topview_image)
        loss = self.loss_fn(outputs, class_index)

        # Metrics computation
        acc = self.accuracy(outputs, class_index)
        auroc_score = self.auroc(outputs, class_index)
        f1 = self.f1_score(outputs, class_index)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auroc", auroc_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        topview_image, _, class_index = batch
        outputs = self(topview_image)
        loss = self.loss_fn(outputs, class_index)

        # Metrics computation
        acc = self.accuracy(outputs, class_index)
        auroc_score = self.auroc(outputs, class_index)
        f1 = self.f1_score(outputs, class_index)

        # Logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_auroc", auroc_score, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)
        return [optimizer], [scheduler]


class SpecimenDataModule(NDPI_DataModule):
    def __init__(self, metadata_file, batch_size=1, num_workers=32):
        super(SpecimenDataModule, self).__init__(metadata_file, batch_size, num_workers)

    def setup(self, stage=None):
        super().setup(stage=stage)


def main():
    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 8
    num_classes = 2  # Number of classes in your dataset

    # Data Module
    data_module = SpecimenDataModule(metadata_file, batch_size)

    # Model
    model = ResNeXtLightningModule(num_classes)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        devices=2 if torch.cuda.is_available() else 0,  # Use 3 GPUs
        accelerator="gpu",  # Ensure that you're using the GPUs
        strategy="ddp",  # Use DistributedDataParallel strategy for multi-GPU
        num_sanity_val_steps=0,  # Skip sanity checks to speed up debugging
        log_every_n_steps=2,
        precision=16  # Use mixed precision for faster training
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
