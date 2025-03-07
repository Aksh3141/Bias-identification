import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from data import dataloader_val
from collections import defaultdict
import numpy as np

# Define the model
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        return self.model(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderClassifier().to(device)
model.load_state_dict(torch.load("gender_classifier.pth", map_location=device))
model.eval()

# Evaluation metrics
correct = 0
total = 0
race_correct = defaultdict(int)
race_total = defaultdict(int)
predictions = defaultdict(list)
ground_truths = defaultdict(list)

# Evaluate model
with torch.no_grad():
    for images, labels, race_labels in dataloader_val:
        images, labels, race_labels = images.to(device), labels.to(device), race_labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            race_correct[race_labels[i].item()] += (predicted[i] == labels[i]).item()
            race_total[race_labels[i].item()] += 1
            predictions[race_labels[i].item()].append(predicted[i].item())
            ground_truths[race_labels[i].item()].append(labels[i].item())

# Overall accuracy
val_acc = 100 * correct / total
print(f"Overall Validation Accuracy: {val_acc:.2f}%")

# Accuracy by race
for race, count in race_total.items():
    acc = 100 * race_correct[race] / count
    print(f"Accuracy for Race {race}: {acc:.2f}%")

# Bias analysis
def demographic_parity(preds, labels):
    positive_rates = {}
    for race, values in preds.items():
        positive_rates[race] = np.mean(np.array(values) == 1)  # Probability of being classified as Female
    return positive_rates

def equal_opportunity(preds, labels):
    true_positive_rates = {}
    for race, values in preds.items():
        mask = np.array(labels[race]) == 1  # Only consider actual females
        if mask.sum() > 0:
            true_positive_rates[race] = np.mean(np.array(values)[mask] == 1)
        else:
            true_positive_rates[race] = None
    return true_positive_rates

# Compute fairness metrics
dp = demographic_parity(predictions, ground_truths)
eo = equal_opportunity(predictions, ground_truths)

print("\nDemographic Parity (Probability of Female classification by race):")
for race, value in dp.items():
    print(f"Race {race}: {value:.4f}")

print("\nEqual Opportunity (True Positive Rate for Female by race):")
for race, value in eo.items():
    if value is not None:
        print(f"Race {race}: {value:.4f}")
    else:
        print(f"Race {race}: No positive samples")
