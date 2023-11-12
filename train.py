import pandas as pd
import numpy as np
import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from preprocess import Preprocess

path = 'train.csv'
train_df = Preprocess(path)

target_col = 'cbg'

# Define sequence length and prediction length
sequence_length = 128
prediction_length = 1  #next sequence prediction

# Split the data into features and target
features = train_df[train_df.columns.difference([target_col])]
target = train_df[target_col]


# Hyperparameters
input_size = features_tensor.shape[1]
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size=32
criterion = nn.MSELoss()


# Convert the data to PyTorch tensors
features_tensor = torch.tensor(features.values, dtype=torch.float32)
target_tensor = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)




# Convert the data to sequences
def create_sequences(features, target, sequence_length, prediction_length):
    sequences = []
    targets = []
    for i in range(len(features) - sequence_length - prediction_length + 1):
        seq = features[i:i + sequence_length]
        label = target[i + sequence_length:i + sequence_length + prediction_length]
        sequences.append(seq)
        targets.append(label)
    return torch.stack(sequences), torch.stack(targets)



# Create sequences and targets
sequences, targets = create_sequences(features_tensor, target_tensor, sequence_length, prediction_length)

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)



# Define a custom dataset class
class OhioT1DMDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'target': self.target[idx]}

# Create instances of the custom dataset for training and testing
train_dataset = OhioT1DMDataset(X_train, y_train)
val_dataset = OhioT1DMDataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




# train function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            features, target = batch['features'], batch['target']
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for batch in tqdm(val_loader, desc='Testing', leave=False):
        features, target = batch['features'], batch['target']
        outputs = model(features)
        loss = criterion(outputs, target)
        test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')


checkpoint_path = 'bgm.pth'
torch.save(model.state_dict(), checkpoint_path)







