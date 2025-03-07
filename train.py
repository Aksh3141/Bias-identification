import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from data import dataloader_train, dataloader_val  # Importing data loaders from data.py
from tqdm import tqdm

# Define the ResNet model for gender classification
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification (Male/Female)
    
    def forward(self, x):
        return self.model(x)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels, _ in progress_bar: 
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)
    
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader_train):.4f}, Train Accuracy: {train_acc:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "gender_classifier.pth")
print("Model saved successfully.")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels, _ in dataloader_val:  # Ignoring race label
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}%")
