import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

val_data = datasets.ImageFolder(
    r"C:\Users\BHAGYASHREE\microplastic_project\Microplastics_dataset\val",
    transform=transform
)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("microplastic_model.pth"))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=val_data.classes))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
