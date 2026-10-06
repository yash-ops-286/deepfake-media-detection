import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from model import DeepfakeDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_sample_dataset_if_empty():
    """Generates synthetic baseline image samples if no data exists."""
    base_dirs = [
        os.path.join("data", "train", "real"),
        os.path.join("data", "train", "fake"),
        os.path.join("data", "val", "real"),
        os.path.join("data", "val", "fake")
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)

    # Check if empty, generate sample benchmark images
    real_train_dir = os.path.join("data", "train", "real")
    if len(os.listdir(real_train_dir)) == 0:
        print("[*] Generating baseline sample training images...")
        for i in range(10):
            # Sample natural-like pattern (Real)
            real_arr = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(real_arr).save(os.path.join("data", "train", "real", f"real_{i}.jpg"))
            Image.fromarray(real_arr).save(os.path.join("data", "val", "real", f"val_real_{i}.jpg"))

            # Sample noise-artifact pattern (Fake)
            fake_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(fake_arr).save(os.path.join("data", "train", "fake", f"fake_{i}.jpg"))
            Image.fromarray(fake_arr).save(os.path.join("data", "val", "fake", f"val_fake_{i}.jpg"))
        print("[✓] Sample baseline data generated.")

def train():
    create_sample_dataset_if_empty()

    train_data = datasets.ImageFolder("data/train", transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    val_data = datasets.ImageFolder("data/val", transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = DeepfakeDetector(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[*] Training pipeline initialized on: {DEVICE}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {total_loss/total:.4f} - Accuracy: {epoch_acc:.2f}%")

    os.makedirs("weights", exist_ok=True)
    save_path = os.path.join("weights", "deepfake_detector.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Fine-tuned model weights saved to {save_path}")

if __name__ == "__main__":
    train()