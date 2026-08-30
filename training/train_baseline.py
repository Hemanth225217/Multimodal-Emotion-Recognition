import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import MELDDataset
from models.fusion_model import MultimodalFusionModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 0.001

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.pt"


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():

        for batch in loader:

            text = batch["text"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Dataset returns:
            # text  -> [utterances, 600]
            # audio -> [utterances, 300]
            # labels -> [utterances]

            # Add batch dimension
            if text.dim() == 2:
                text = text.unsqueeze(0)

            if audio.dim() == 2:
                audio = audio.unsqueeze(0)

            if labels.dim() == 1:
                labels = labels.unsqueeze(0)

            # Forward pass
            output = model(text, audio)

            # Output:
            # [batch, utterances, 7]

            loss = criterion(
                output.reshape(-1, 7),
                labels.reshape(-1)
            )

            total_loss += loss.item()

            predictions = output.argmax(dim=-1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_items += labels.numel()

    average_loss = total_loss / len(loader)

    accuracy = total_correct / total_items

    return average_loss, accuracy


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MELD BASELINE TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = MELDDataset("train")

    # IMPORTANT:
    # dataset.py accepts "val", not "dev"
    val_dataset = MELDDataset("val")

    print(
        "Training dialogues:",
        len(train_dataset)
    )

    print(
        "Validation dialogues:",
        len(val_dataset)
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    print("\nCreating model...")

    model = MultimodalFusionModel(
        text_dim=600,
        audio_dim=300,
        hidden_dim=256,
        num_classes=7
    ).to(DEVICE)

    print(model)

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_items = 0

        print("\n" + "=" * 60)
        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )
        print("=" * 60)

        for step, batch in enumerate(train_loader):

            # --------------------------------------------------
            # Get data
            # --------------------------------------------------

            text = batch["text"].to(DEVICE)

            audio = batch["audio"].to(DEVICE)

            labels = batch["labels"].to(DEVICE)

            # --------------------------------------------------
            # Add batch dimension
            # --------------------------------------------------

            if text.dim() == 2:
                text = text.unsqueeze(0)

            if audio.dim() == 2:
                audio = audio.unsqueeze(0)

            if labels.dim() == 1:
                labels = labels.unsqueeze(0)

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            output = model(
                text,
                audio
            )

            # output:
            # [batch, utterances, 7]

            # labels:
            # [batch, utterances]

            # --------------------------------------------------
            # Calculate loss
            # --------------------------------------------------

            loss = criterion(
                output.reshape(-1, 7),
                labels.reshape(-1)
            )

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # --------------------------------------------------
            # Calculate accuracy
            # --------------------------------------------------

            predictions = output.argmax(
                dim=-1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_items += labels.numel()

            total_loss += loss.item()

            # --------------------------------------------------
            # Show progress
            # --------------------------------------------------

            if (step + 1) % 100 == 0:

                current_accuracy = (
                    total_correct /
                    total_items
                )

                print(
                    f"Step {step + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f} "
                    f"| Accuracy: {current_accuracy:.4f}"
                )

        # --------------------------------------------------
        # Training statistics
        # --------------------------------------------------

        train_loss = (
            total_loss /
            len(train_loader)
        )

        train_accuracy = (
            total_correct /
            total_items
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        print("\nRESULTS")
        print("-" * 40)

        print(
            f"Train Loss:     {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:       {val_loss:.4f}"
        )

        print(
            f"Val Accuracy:   {val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("\n✓ Best model saved!")

            print(
                "Path:",
                MODEL_PATH
            )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best validation accuracy:",
        best_accuracy
    )

    print(
        "Model saved at:",
        MODEL_PATH
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()