import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, AUROC
from DZSpecimenClf import DZSpecimenClf
from datamodule import NDPI_DataModule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SpecimenClassifier(nn.Module):
    def __init__(self, N, num_classes=2, patch_size=224):
        super(SpecimenClassifier, self).__init__()
        self.model = DZSpecimenClf(N, num_classes=num_classes, patch_size=patch_size)

    def forward(self, topview_image_tensor, search_view_indexibles):
        return self.model(topview_image_tensor, search_view_indexibles)


def train(
    model,
    dataloader,
    optimizer,
    loss_fn,
    accuracy,
    auroc,
    f1_score,
    device,
    writer,
    epoch,
):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_auroc = 0
    total_f1 = 0

    for batch_idx, batch in enumerate(dataloader):
        topview_image, search_view_indexible, class_index = batch
        topview_image, class_index = topview_image.to(device), class_index.to(device)

        optimizer.zero_grad()

        outputs = model(topview_image, search_view_indexible)
        loss = loss_fn(outputs, class_index)

        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, class_index)
        auroc_score = auroc(outputs, class_index)
        f1 = f1_score(outputs, class_index)

        total_loss += loss.item()
        total_accuracy += acc.item()
        total_auroc += auroc_score.item()
        total_f1 += f1.item()

        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}"
            )

        # Logging to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("train_loss", loss.item(), global_step)
        writer.add_scalar("train_acc", acc.item(), global_step)
        writer.add_scalar("train_auroc", auroc_score.item(), global_step)
        writer.add_scalar("train_f1", f1.item(), global_step)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_auroc = total_auroc / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)

    print(
        f"Train Epoch: {epoch} Avg Loss: {avg_loss:.6f} Avg Acc: {avg_accuracy:.6f} Avg AUROC: {avg_auroc:.6f} Avg F1: {avg_f1:.6f}"
    )


def validate(
    model,
    dataloader,
    loss_fn,
    accuracy,
    auroc,
    f1_score,
    device,
    writer,
    epoch,
    best_metric,
    save_path,
):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_auroc = 0
    total_f1 = 0

    with torch.no_grad():
        for batch in dataloader:
            topview_image, search_view_indexible, class_index = batch
            topview_image, class_index = topview_image.to(device), class_index.to(
                device
            )

            outputs = model(topview_image, search_view_indexible)
            loss = loss_fn(outputs, class_index)

            acc = accuracy(outputs, class_index)
            auroc_score = auroc(outputs, class_index)
            f1 = f1_score(outputs, class_index)

            total_loss += loss.item()
            total_accuracy += acc.item()
            total_auroc += auroc_score.item()
            total_f1 += f1.item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_auroc = total_auroc / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)

    print(
        f"Validation Epoch: {epoch} Avg Loss: {avg_loss:.6f} Avg Acc: {avg_accuracy:.6f} Avg AUROC: {avg_auroc:.6f} Avg F1: {avg_f1:.6f}"
    )

    # Logging to TensorBoard
    writer.add_scalar("val_loss", avg_loss, epoch)
    writer.add_scalar("val_acc", avg_accuracy, epoch)
    writer.add_scalar("val_auroc", avg_auroc, epoch)
    writer.add_scalar("val_f1", avg_f1, epoch)

    # Check if this is the best performance so far
    current_metric = (
        avg_accuracy  # or avg_auroc, avg_f1 depending on which metric you prefer
    )
    if current_metric > best_metric:
        print(f"New best model found! Saving to {save_path}")
        torch.save(model.state_dict(), save_path)
        best_metric = current_metric

    return best_metric


def main():
    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 16
    N = 8  # Example value
    patch_size = 224
    num_classes = 2  # Number of classes in your dataset
    save_path = "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate dataset and dataloaders
    data_module = NDPI_DataModule(metadata_file, batch_size, num_workers=64)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Instantiate model, loss, optimizer, and metrics
    model = SpecimenClassifier(N, patch_size=patch_size, num_classes=num_classes).to(
        device
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)
    f1_score = F1Score(num_classes=num_classes, task="multiclass").to(device)
    auroc = AUROC(num_classes=num_classes, task="multiclass").to(device)

    # TensorBoard logger
    writer = SummaryWriter("runs/my_model")

    # Training loop
    num_epochs = 50
    best_metric = 0.0  # Initialize best metric
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"Epoch {epoch} training ... ")
        train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            accuracy,
            auroc,
            f1_score,
            device,
            writer,
            epoch,
        )

        print("Validation")
        best_metric = validate(
            model,
            val_loader,
            loss_fn,
            accuracy,
            auroc,
            f1_score,
            device,
            writer,
            epoch,
            best_metric,
            save_path,
        )
        scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()
