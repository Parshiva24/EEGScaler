import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import LeaveOneOut

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import direction_utils as utils
import constants
import random

learning_rate=constants.learning_rate   #1e-4
num_epochs = constants.num_epochs   #1000
patience = constants.patience   #30  # Number of epochs to wait before stopping if no improvement
min_delta = constants.min_delta   #1e-3 # Minimum change to qualify as improvement

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED=constants.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')

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
    mdl_filename = os.path.join(constants.mdl_dir, 'trained_model_checkpoint.pth')

    if not Tuning and os.path.exists(mdl_filename):
        os.remove(mdl_filename)
        print('deleted existing model')
    # else:
        # print('No saved model')


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
                torch.save(model.state_dict(), mdl_filename)  # Save best model
                if verbose:
                    print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}')

            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1} with best validation accuracy: {best_val_acc:.4f}')
                break

        else:
            if epoch > constants.min_epochs:
                # Early stopping check
                if val_acc > best_val_acc + min_delta:  # Check if validation Acc improved
                    best_val_acc = val_acc
                    early_stop_counter = 0  # Reset early stop counter
                    # torch.save(model.state_dict(), 'trained_model_checkpoint.pth')  # Save best model4
                    torch.save(model.state_dict(), mdl_filename)
                    if verbose:
                        print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}')

                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience and best_val_acc > constants.best_val_acc:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1} with best validation accuracy: {best_val_acc:.4f}')
                    break
    
    model.load_state_dict(torch.load(os.path.join(constants.mdl_dir, 'trained_model_checkpoint.pth')))

    return model
 
# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy*100

def convert_to_tensor(X, Y, batch_size=32, train_flag=False):
    
    Xbase = utils.baseline_correction(X)
    Xfilt = utils.bandpass_filtering(Xbase)

    if train_flag:
        Xfilt, Y = utils.data_augmentation_timeshift(Xfilt, Y)

    X_tensor = torch.tensor(Xfilt, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2).to(device)  # Shape: (batch_size, 1, 27, 2500)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)  # Use long for classification

    torch_data = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(dataset=torch_data, batch_size=batch_size, shuffle=train_flag)


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
        self.electrode_scale = ElectrodeScaler(Chan, Samples, reduction=constants.el_scaler_red_rate)
    
        #TemporalScaler
        self.sample_scale = TemporalScaler(Chan, Samples, reduction=constants.sample_scaler_red_rate)

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

        x = self.pool2(x)
        x = self.dropout(x)

        # Flatten and Dense
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)


def augmented_dataset_preparation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data_dir = os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')
    train_ratio = constants.train_ratio

    subids = list(range(1, 15))
    loo = LeaveOneOut()

    for train_subs, test_subs in loo.split(subids):
        train_subids = [subids[i] for i in train_subs]
        test_subids = [subids[i] for i in test_subs] 

        filepath = os.path.join(constants.data_dir, 'data', f'augmented_data_testsub_S{test_subids[0]:02d}.pt')
        print(filepath)

        train_data, train_label, test_data, test_label = utils.speed_dataset(train_subids, test_subids, base_path=constants.data_dir)
        eeg_train, eeg_val, label_train, label_val = train_test_split(train_data, train_label, 
                                                                train_size=train_ratio, random_state=42, shuffle=True)
        
        train_loader = convert_to_tensor(eeg_train, label_train, train_flag=True)
        val_loader = convert_to_tensor(eeg_val, label_val, train_flag=False)

        torch_train_dataset = train_loader.dataset
        Xtr, Ytr = torch_train_dataset.tensors

        torch_val_dataset = val_loader.dataset
        Xval, Yval = torch_val_dataset.tensors
        torch.save({
            'Xtr': Xtr.cpu(),
            'Ytr': Ytr.cpu(),
            'Xval': Xval.cpu(),
            'Yval': Yval.cpu()}, 
            filepath)
        
        # Fine tuned the scaling layers only to report the performance. 
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold=1

        for train_idx, test_idx in kf.split(test_data):
            # Split into training and test sets
            Xtr, Xte = test_data[train_idx], test_data[test_idx]
            Ytr, Yte = test_label[train_idx], test_label[test_idx]

            Xtrain, Xval, Ytrain, Yval = train_test_split(Xtr, Ytr, train_size=train_ratio, random_state=42, shuffle=True)
            
            train_loader = convert_to_tensor(Xtrain, Ytrain, batch_size=constants.batch_size, train_flag=True)
            val_loader = convert_to_tensor(Xval, Yval, batch_size=constants.batch_size, train_flag=False)
            test_loader = convert_to_tensor(Xte, Yte, batch_size=Yte.size, train_flag=False)

            torch_train_dataset = train_loader.dataset
            Xtr, Ytr = torch_train_dataset.tensors

            torch_val_dataset = val_loader.dataset
            Xval, Yval = torch_val_dataset.tensors

            torch_test_dataset = test_loader.dataset
            Xte, Yte = torch_test_dataset.tensors

            filepath = os.path.join(constants.data_dir, 'data',  f'augmented_data_fold{fold}_S{test_subids[0]:02d}.pt')
            print(f'  ...{filepath}')

            torch.save({
                'Xtr': Xtr.cpu(),
                'Ytr': Ytr.cpu(),
                'Xval': Xval.cpu(),
                'Yval': Yval.cpu(), 
                'Xte' : Xte.cpu(), 
                'Yte':Yte.cpu()}, 
                filepath)
            
            fold += 1
    return 




if __name__=='__main__':
    # Example Usage
    # batch, filters, height, width = 32, 1, 27, 2000
    # x = torch.randn(batch, filters, height, width)

    # electrodes, samples = 27, 2000
    # x = torch.randn(electrodes, samples)

    # model = EEGScaler(nb_classes=2, Chan=27, Samples=2000)

    # # model = TemporalScaler(height=height, width=width)
    # scaled_x = model(x)
    # print("Output shape:", scaled_x.shape)  # Expected: (32, 1, 27, 2000)

    augmented_dataset_preparation()