import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1, AUROC
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.models import resnext50_32x4d
from torch.optim.lr_scheduler import CosineAnnealingLR
from datamodule import NDPI_DataModule

class ResNeXtClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        # Initialize the ResNeXt model
        self.model = resnext50_32x4d(pretrained=True)
        # Modify the final layer to output the correct number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        
        self.learning_rate = learning_rate
        # Metrics
        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=num_classes, average="macro")
        self.auroc = AUROC(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        top_view_images, _, class_indices = batch
        logits = self(top_view_images)
        loss = F.cross_entropy(logits, class_indices)

        preds = torch.argmax(logits, dim=1)
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.accuracy(preds, class_indices), on_step=True, on_epoch=True)
        self.log('train_f1', self.f1(preds, class_indices), on_step=True, on_epoch=True)
        self.log('train_auroc', self.auroc(logits, class_indices), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        top_view_images, _, class_indices = batch
        logits = self(top_view_images)
        loss = F.cross_entropy(logits, class_indices)

        preds = torch.argmax(logits, dim=1)
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, class_indices), on_step=False, on_epoch=True)
        self.log('val_f1', self.f1(preds, class_indices), on_step=False, on_epoch=True)
        self.log('val_auroc', self.auroc(logits, class_indices), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        top_view_images, _, class_indices = batch
        logits = self(top_view_images)
        loss = F.cross_entropy(logits, class_indices)

        preds = torch.argmax(logits, dim=1)
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.accuracy(preds, class_indices), on_step=False, on_epoch=True)
        self.log('test_f1', self.f1(preds, class_indices), on_step=False, on_epoch=True)
        self.log('test_auroc', self.auroc(logits, class_indices), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

# Load your data
data_module = NDPI_DataModule(metadata_file='path_to_metadata.csv', batch_size=32, num_workers=32)

# Set up trainer
trainer = Trainer(
    max_epochs=20,
    gpus=3,  # Use 3 GPUs
    accelerator="gpu",  # Enable GPU usage
    strategy="ddp",  # Use DistributedDataParallel for multi-GPU training
    callbacks=[LearningRateMonitor(logging_interval='step')],
    precision=16  # Use mixed precision for faster training
)

# Initialize your model
model = ResNeXtClassifier(num_classes=2)

# Train the model
trainer.fit(model, datamodule=data_module)

# Test the model
trainer.test(model, datamodule=data_module)
