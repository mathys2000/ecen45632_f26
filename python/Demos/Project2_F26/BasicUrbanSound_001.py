# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ecen45632 [conda env:ecen45632]
#     language: python
#     name: conda-env-ecen45632-ecen45632
# ---

# %% [markdown]
# # Basic Urban Sound 001
#
# Basic UrbanSound Classifier 001, using PyTorch

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# %%
# Urban sound classes
CLASS_MAPPING = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]


# %%
class UrbanSoundDataset(Dataset):
    """
    PyTorch Dataset for UrbanSound8K
    
    Args:
        annotations_file: Path to metadata CSV file
        audio_dir: Root directory containing fold subdirectories
        transformation: Audio transformation to apply (e.g., Spectrogram)
        target_sample_rate: Sample rate to resample audio to
        num_samples: Number of samples to pad/truncate to
        fold: List of folds or specific fold to load (None for all folds)
    """
    def __init__(self, 
                 annotations_file,
                 audio_dir,
                 transformation=None,
                 target_sample_rate=22050,
                 num_samples=44100,
                 fold=None):
        self.annotations = pd.read_csv(annotations_file)
        
        # Filter by fold or list of folds if specified
        if fold is not None:
            if isinstance(fold, list) or isinstance(fold, np.ndarray):
                self.annotations = self.annotations[self.annotations['fold'].isin(fold)]
            else:
                self.annotations = self.annotations[self.annotations['fold'] == fold]
        
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get audio file path
        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        
        # Load audio
        signal, sr = torchaudio.load(audio_sample_path)
        
        # Resample if necessary
        signal = self._resample_if_necessary(signal, sr)
        
        # Mix down to mono if necessary
        signal = self._mix_down_if_necessary(signal)
        
        # Pad or truncate to fixed length
        signal = self._pad_or_truncate(signal)
        
        # Apply transformation
        if self.transformation:
            signal = self.transformation(signal)
        
        return signal.squeeze(0), label
    
    def _get_audio_sample_path(self, idx):
        fold = f"fold{self.annotations.iloc[idx]['fold']}"
        filename = self.annotations.iloc[idx]['slice_file_name']
        path = os.path.join(self.audio_dir, fold, filename)
        return path
    
    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx]['classID']
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _pad_or_truncate(self, signal):
        length = signal.shape[1]
        if length > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif length < self.num_samples:
            pad_length = self.num_samples - length
            signal = torch.nn.functional.pad(signal, (0, pad_length))
        return signal



# %%
# Parameters
BATCH_SIZE = 32
N_FRQ = 64

ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 44100   # 2 sec at 22050 Hz

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
# Instantiate datasets and dataloaders
    
# Define transformation (Spectrogram)
spectrogram_transform = T.Spectrogram(
        n_fft=128,
        hop_length=512,
        normalized=True
)

# Create dataset for fold 1 (training)
train_dataset = UrbanSoundDataset(
    annotations_file=ANNOTATIONS_FILE,
    audio_dir=AUDIO_DIR,
    transformation=spectrogram_transform,
    target_sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    fold=1
)

# Create dataset for fold 9 (validating)
val_dataset = UrbanSoundDataset(
    annotations_file=ANNOTATIONS_FILE,
    audio_dir=AUDIO_DIR,
    transformation=spectrogram_transform,
    target_sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    fold=9
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# %%
# Define Convolutional Neural Network
class UrbanSoundCNN(nn.Module):
    """CNN for urban sound classification using filterbank features"""
    def __init__(self, n_frq=64, n_classes=10, dropout=0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3 , padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, n_frq, time_frames)
        Returns:
            logits: (batch, n_classes)
        """
        # Add channel dimension
        x = x.unsqueeze(1)   # (batch, 1, n_frq, time_frames)

        # Convolutional  blocks
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# %%
# Train and validate function
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Training loop with comprehensive tracking"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            #print(f"Features shape {features.shape}, Labels shape {labels.shape}")
            
            optimizer.zero_grad()
            outputs = model(features)
            #print(f"Outputs len {len(outputs)}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs



# %%
# Initialize the model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
    
# Initialize model
model = UrbanSoundCNN(n_frq=N_FRQ, n_classes=len(CLASS_MAPPING))
model = model.to(device)
    
print(f'\nModel Architecture:')
print(model)
print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')


# %%
# Train the model
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=20)

# %%
print(f"train_losses {train_losses}")
print(f"val_losses {val_losses}")
print(f"train_accs {train_accs}")
print(f"val_accs {val_accs}")

# %%

# %%
