import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_cnn_module():
    root = Path(__file__).resolve().parent
    cnn_path = root / "Flask Deployed App" / "CNN.py"
    if not cnn_path.exists():
        raise FileNotFoundError(f"Could not find CNN.py at {cnn_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("cnn_module", str(cnn_path))
    cnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_module)
    return cnn_module


def get_data_loaders(data_root: Path, batch_size: int, num_workers: int, max_train: int, max_valid: int, max_test: int):
    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    def resolve_nested_folder(folder: Path) -> Path:
        current = folder
        while current.exists() and current.is_dir():
            subdirs = [p for p in current.iterdir() if p.is_dir()]
            if len(subdirs) == 1 and not any(p.is_file() for p in current.iterdir()):
                current = subdirs[0]
                continue
            break
        return current

    train_dir = resolve_nested_folder(data_root / "train")
    valid_dir = resolve_nested_folder(data_root / "valid")
    test_dir = resolve_nested_folder(data_root / "test")

    if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Dataset directories not found. Expected 'train', 'valid', and 'test' under the dataset root."
        )

    train_dataset = ImageFolder(str(train_dir), transform=transform)
    valid_dataset = ImageFolder(str(valid_dir), transform=transform)

    def has_class_folder(folder: Path) -> bool:
        return any(p.is_dir() for p in folder.iterdir())

    if has_class_folder(test_dir):
        test_dataset = ImageFolder(str(test_dir), transform=transform)
    else:
        test_dataset = None
        print(f"Warning: no class folders found in test dataset at {test_dir}. Skipping test loader.")

    def maybe_subset(dataset, max_samples):
        if dataset is None:
            return None
        if max_samples and 0 < max_samples < len(dataset):
            from torch.utils.data import Subset

            return Subset(dataset, list(range(max_samples)))
        return dataset

    train_dataset = maybe_subset(train_dataset, max_train)
    valid_dataset = maybe_subset(valid_dataset, max_valid)
    test_dataset = maybe_subset(test_dataset, max_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    return train_loader, valid_loader, test_loader, train_dataset


def accuracy(output, targets):
    preds = output.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, targets) * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count, running_acc / count


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, targets) * inputs.size(0)
            count += inputs.size(0)

    return running_loss / count, running_acc / count


def main():
    parser = argparse.ArgumentParser(description="Train the plant disease CNN model.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Model") / "Dataset",
        help="Root path to the dataset containing train/valid/test folders.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 on Windows for compatibility.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If set, only use this many training samples for a quick demo run.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=0,
        help="If set, only use this many validation samples for a quick demo run.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="If set, only use this many test samples for a quick demo run.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("Flask Deployed App") / "plant_disease_model_1_latest.pt",
        help="Path to save the trained model checkpoint.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    cnn_module = load_cnn_module()
    CNN = cnn_module.CNN
    print("Loading dataset from", args.data_root)
    train_loader, valid_loader, test_loader, train_dataset = get_data_loaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        args.max_train_samples,
        args.max_valid_samples,
        args.max_test_samples,
    )

    if len(getattr(train_dataset, 'dataset', [])):
        print("Using subset of training samples", len(train_dataset))

    dataset_classes = (
        train_dataset.dataset.classes
        if hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "classes")
        else train_dataset.classes
    )
    num_classes = len(dataset_classes)
    print("Dataset classes:", num_classes)
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved checkpoint: {args.save_path}")

    test_loss = None
    test_acc = None
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}"
        )
    else:
        print("Skipping test evaluation because no labeled test classes were found.")

    stats_file = Path("train_history.json")
    stats = {
        "classes": dataset_classes,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    stats_file.write_text(json.dumps(stats, indent=2))
    print(f"Saved training history to {stats_file}")

    mapping_path = Path("Flask Deployed App") / "class_mapping.json"
    mapping = {str(i): label for i, label in enumerate(dataset_classes)}
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Saved class mapping to {mapping_path}")


if __name__ == "__main__":
    main()
