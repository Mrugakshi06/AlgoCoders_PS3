import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data transforms ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# --- Load datasets ---
train_data = datasets.ImageFolder(
    r"C:\Users\BHAGYASHREE\microplastic_project\Microplastics_dataset\train",
    transform=transform
)

val_data = datasets.ImageFolder(
    r"C:\Users\BHAGYASHREE\microplastic_project\Microplastics_dataset\val",
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

print("Train classes:", train_data.classes)
print("Val classes:", val_data.classes)
print("Train samples:", len(train_data))
print("Val samples:", len(val_data))

# --- Model setup ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # updated syntax
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes
model = model.to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training loop ---
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

# --- Save model ---
torch.save(model.state_dict(), "microplastic_model.pth")
print("Training complete. Model saved as microplastic_model.pth")
