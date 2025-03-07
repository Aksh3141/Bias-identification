import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Convert categorical labels to numerical encoding
        self.race_map = {race: idx for idx, race in enumerate(self.df['race'].unique())}
        self.gender_map = {'Male': 0, 'Female': 1}
        
        self.df['race_label'] = self.df['race'].map(self.race_map)
        self.df['gender_label'] = self.df['gender'].map(self.gender_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['file'])
        image = Image.open(img_name).convert('RGB')
        gender_label = self.df.iloc[idx]['gender_label']
        race_label = self.df.iloc[idx]['race_label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(gender_label, dtype=torch.long), torch.tensor(race_label, dtype=torch.long)

# transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset paths
csv_train_file = "/home/ankur/Desktop/GSOC/fairface_label_train.csv"
image_train_dir = "/home/ankur/Desktop/GSOC/fairface-img-margin125-trainval"

csv_val_file = "/home/ankur/Desktop/GSOC/fairface_label_val.csv"
image_val_dir = "/home/ankur/Desktop/GSOC/fairface-img-margin125-trainval"

# Training set
dataset_train = FairFaceDataset(csv_train_file, image_train_dir, transform)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)

# Validation set
dataset_val = FairFaceDataset(csv_val_file, image_val_dir, transform)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
