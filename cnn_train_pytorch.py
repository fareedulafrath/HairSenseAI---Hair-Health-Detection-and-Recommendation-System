import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths & settings
DATA_DIR = "hair_dataset"
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Datasets & loaders
train_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
train_size = int(0.8 * len(train_ds))
val_size = len(train_ds) - train_size
train_data, val_data = torch.utils.data.random_split(train_ds, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# CNN architecture
class HairCNN(nn.Module):
    def __init__(self):
        super(HairCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Instantiate model, loss, optimizer
model = HairCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        total += labels.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    val_acc = 0
    # Validation
    model.eval()
    with torch.no_grad():
        val_total, val_correct = 0, 0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total

    print(f"Epoch {epoch} — Loss: {loss_sum/len(train_loader):.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

# Save model
torch.save(model.state_dict(), "cnn_hair_model.pth")
print("✅ CNN model saved as cnn_hair_model.pth")
