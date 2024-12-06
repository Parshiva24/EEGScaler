import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import direction_utils as utils
import constants

learning_rate=constants.learning_rate   #1e-4
num_epochs = constants.num_epochs   #1000
patience = constants.patience   #30  # Number of epochs to wait before stopping if no improvement
min_delta = constants.min_delta   #1e-3 # Minimum change to qualify as improvement

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0)


def model_evaluation(model, val_loader):
    model.eval()
    # Loss and Accuracy
    val_loss = 0.0
    val_acc = 0.0

    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU if needed
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # se_electrode1 = model.electrode_weights_layer1[-1].tolist()
            # se_electrode2 = model.electrode_weights_layer2.cpu().tolist()

            # se1_scales = model.se1_scales.cpu().tolist()
            # se2_scales = model.se2_scales.cpu().tolist()
            # se3_scales = model.se3_scales.cpu().tolist()
    
            loss = criterion(outputs, targets.squeeze())
            val_loss += loss.item()
            val_acc += calculate_accuracy(outputs, targets.squeeze())

    # Calculate average validation loss for this epoch
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # assigned_ranks = {'electrodes_se1':se_electrode1, 
    #                   'electrodes_se2':se_electrode2,
    #                   'filter_se1': se1_scales, 
    #                   'filter_se2': se2_scales, 
    #                   'filter_se3': se3_scales 
    #                   }
    assigned_ranks = None

    return val_loss, val_acc, assigned_ranks

def model_training(model, train_loader, val_loader, Tuning=False, verbose=False):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU if needed

    # Training with Early Stopping
    best_val_acc = 0
    early_stop_counter = 0

    # Placeholder for training and validation loss history
    train_losses = []
    val_losses = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # val_loss, val_acc, el_rank1, el_rank2 = model_evaluation(model, val_loader)
        val_loss, val_acc, _ = model_evaluation(model, val_loader)

        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%')
        
        if Tuning:
            # Early Stopping
            # Early stopping check
            if val_acc > best_val_acc + min_delta:  # Check if validation Acc improved
                best_val_acc = val_acc
                early_stop_counter = 0  # Reset early stop counter
                torch.save(model.state_dict(), 'trained_model_checkpoint.pth')  # Save best model
                if verbose:
                    print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}')

            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1} with best validation accuracy: {best_val_acc:.4f}')
                break

        else:
            if epoch > 100:
                # Early stopping check
                if val_acc > best_val_acc + min_delta:  # Check if validation Acc improved
                    best_val_acc = val_acc
                    early_stop_counter = 0  # Reset early stop counter
                    torch.save(model.state_dict(), 'trained_model_checkpoint.pth')  # Save best model
                    if verbose:
                        print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}')

                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience and best_val_acc > 70.00:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1} with best validation accuracy: {best_val_acc:.4f}')
                    break
    
    return model
 
# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy*100

def convert_to_tensor(X, Y, batch_size=32, shuffle=False):
    
    Xbase = utils.baseline_correction(X)
    Xfilt = utils.bandpass_filtering(Xbase)

    X_tensor = torch.tensor(Xfilt, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2).to(device)  # Shape: (batch_size, 1, 27, 2500)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)  # Use long for classification

    torch_data = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(dataset=torch_data, batch_size=batch_size, shuffle=shuffle)


    return data_loader

class ElectrodeScaler(nn.Module):
    def __init__(self, height, width, reduction=3):
        super().__init__()
        self.height = height
        self.width = width

        self.shared_mlp = nn.Sequential(
                            nn.Linear(height, height//reduction, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(height//reduction, height, bias=False)
                            )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = torch.mean(x, dim=3, keepdim=True) # Mean poolin over samples (batch, 1, electrodes, 1)
        x2 = torch.max(x, dim=3, keepdim=True)[0] # Max Pooling

        # Reshape the data
        batch, filters, height, _ = x.shape
        x1 = x1.view(batch*filters, height) # Reshape 
        x2 = x2.view(batch*filters, height)

        # Apply Shared MLP to Pooled Data
        out1 = self.shared_mlp(x1)
        out2 = self.shared_mlp(x2)

        combined_out = self.sigmoid(out1+out2)
        combined_out = combined_out.view(batch, filters, height, 1)

        # scaled_x = x*combined_out.expand_as(x)


        return combined_out

class TemporalScaler(nn.Module):
    def __init__(self, height, width, reduction=16):
        super().__init__()
        self.height = height
        self.width = width

        self.shared_mlp = nn.Sequential(
                            nn.Linear(width, width//reduction, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(width//reduction, width, bias=False)
                            )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # Pooling Operation
        x1 = torch.mean(x, dim=2, keepdim=True) # Mean Pooling over electrodes -> (32, 1, 1, 2000)
        x2 = torch.max(x, dim=2, keepdim=True)[0] # Max Value alone is used

        # Reshape the data
        batch, filters, _, width = x.shape
        x1 = x1.view(batch*filters, width) # Reshape 
        x2 = x2.view(batch*filters, width)


        # Apply Shared MLP to Pooled Data
        out1 = self.shared_mlp(x1)
        out2 = self.shared_mlp(x2)

        combined_out = self.sigmoid(out1+out2)
        combined_out = combined_out.view(batch, filters, 1, width)

        # scaled_x = x*combined_out.expand_as(x)

        return combined_out

class EEGScaler(nn.Module):
    def __init__(self, nb_classes, Chan=27, Samples=2000, dropoutRate=0.5, 
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGScaler, self).__init__()

        # Handle dropout type
        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout(dropoutRate)
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout.')
        
        #ElectrodeScaler
        self.electrode_scale = ElectrodeScaler(Chan, Samples, reduction=3)
    
        #TemporalScaler
        self.sample_scale = TemporalScaler(Chan, Samples, reduction=16)

        #Depthwise CNN
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        self.depthwiseConv = nn.Conv2d(F1, F1*D, (Chan, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1*D)

        self.pool1 = nn.AvgPool2d((1, 4))

        #Spatial CNN
        self.separableConv = nn.Conv2d(F1*D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))

        # Flatten and Dense
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // (4 * 8)), nb_classes)
        self.norm_constraint = nn.utils.weight_norm(self.dense)
    
    def forward(self, x):

        electrode_scales = self.electrode_scale(x)
        x = x*electrode_scales.expand_as(x)
        
        sample_scales = self.sample_scale(x)
        x = x*sample_scales.expand_as(x)

        # Convolution Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = x*electrode_scales.expand_as(x)
        x = x*sample_scales.expand_as(x)
       
        # Convolution Block 2
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = x*sample_scales.expand_as(x)

        x = self.pool1(x)
        x = self.dropout(x)

        # Convolution Block 3
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        # x = x*sample_scales.expand_as(x)

        x = self.pool2(x)
        x = self.dropout(x)

        # Flatten and Dense
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)


if __name__=='__main__':
    # Example Usage
    # batch, filters, height, width = 32, 1, 27, 2000
    # x = torch.randn(batch, filters, height, width)

    electrodes, samples = 27, 2000
    x = torch.randn(electrodes, samples)

    model = EEGScaler(nb_classes=2, Chan=27, Samples=2000)

    # model = TemporalScaler(height=height, width=width)
    scaled_x = model(x)
    print("Output shape:", scaled_x.shape)  # Expected: (32, 1, 27, 2000)