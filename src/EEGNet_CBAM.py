import numpy as np
import pandas as pd
import os
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import direction_utils as utils
import direction_learning_utils as train_utils
import constants

torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0)

def model_using_calibration_data(verbose=False):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = constants.data_dir
    print(data_dir)

    # project_dir = os.path.dirname(os.getcwd())
    # parent_dir = os.path.dirname(project_dir)
    # data_dir = os.path.join(parent_dir, 'MIDecoding_SENet')
    # print(data_dir)

    
    batch_size=constants.batch_size
    fs = constants.fs
    train_ratio = constants.train_ratio
    sub = constants.train_sub_indx #Till subject 7 (starting from 0), only calibration sessions are conducted

    # Xtr, Ytr = create_dataset(sub, base_path=parent_dir)
    Xtr, Ytr = utils.calib_sess_dataset(base_path=data_dir)
    # Creating train-validation split

    eeg_train, eeg_val, label_train, label_val = train_test_split(Xtr, Ytr, 
                                                                train_size=train_ratio, random_state=42, shuffle=True)
    
    train_loader = train_utils.convert_to_tensor(eeg_train, label_train)
    val_loader = train_utils.convert_to_tensor(eeg_val, label_val)

    model = train_utils.EEGScaler(nb_classes=2, Chan=27, Samples=2000).to(device)

    trained_model = train_utils.model_training(model, train_loader, val_loader, verbose=verbose)

    out_dir = os.path.join(constants.mdl_dir, 'baseElectrodeScaler.pth')
    torch.save(trained_model.state_dict(), out_dir)

    return trained_model

def evaluation_on_online_data(model_file):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = constants.data_dir
    mdl_dir = os.path.join(constants.mdl_dir, model_file)

    model = train_utils.EEGScaler(nb_classes=2, Chan=27, Samples=2000).to(device)
    model.load_state_dict(torch.load(f'{mdl_dir}.pth'))

    # hook_handle = model.se_electrode1.sigmoid.register_forward_hook(forward_hook)

    online_perf = dict()

    for sub in range(8, 21):
        Xtr, Ytr, Xte, Yte = utils.online_sess_dataset(sub, base_path=data_dir)
        batch_size = Yte.size
        # print(batch_size)
        test_loader = train_utils.convert_to_tensor(Xte, Yte, batch_size=batch_size)

        test_loss, test_acc, _ = train_utils.model_evaluation(model, test_loader)
        

        online_perf[f'Sub{sub:02d}'] = test_acc
        
        print(f'Subject: {sub:02d}, Test Accuracy: {test_acc:.2f}%')

    print(f'Average Accuracy: {np.mean(list(online_perf.values()))}')
    print(list(online_perf.values()))

    return online_perf
    
# model fine tuning params
def model_fine_tuning_params(model_file, electrode_scale=False, sample_scale=False, dense=True):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mdl_dir = os.path.join(constants.mdl_dir, model_file)
    model = train_utils.EEGScaler(nb_classes=2, Chan=27, Samples=2000).to(device)
    model.load_state_dict(torch.load(f'{mdl_dir}.pth'))

    if dense or electrode_scale or sample_scale:
        for param in model.parameters():
            param.requires_grad = False

        if electrode_scale:
            print(f'Electrode Scaling are fine-tuned')
            for param in model.electrode_scale.parameters():
                param.requires_grad = True  # Unfreeze  layer
        
        if sample_scale:
            print(f'Temporal Sample Scales are fine-tuned')
            for param in model.sample_scale.parameters():
                param.requires_grad = True  # Unfreeze  layer
        
        if dense:
            print(f'Dense Layer is fine-tuned')
            for param in model.dense.parameters():
                param.requires_grad = True  # Unfreeze  layer
    
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=constants.learning_rate)

    return model, optimizer
        

# Subject Specific Fine Tuning of Model
def subject_specific_fine_tuning(electrode_scaling=True, sample_scale=True, dense=True):
    data_dir = constants.data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Xcalib, Ycalib = utils.calib_sess_dataset(base_path=data_dir)

    train_ratio = constants.train_ratio
    batch_size = constants.batch_size

    online_perf = dict()
    model_file = 'baseElectrodeScaler'
    for sub in range(8, 21):
        Xtr, Ytr, Xte, Yte = utils.online_sess_dataset(sub, base_path=data_dir)
        eeg_train, eeg_val, label_train, label_val = train_test_split(Xtr, Ytr, 
                                                                train_size=train_ratio, random_state=42, shuffle=True)
        Xtrain = eeg_train
        Ytrain = label_train
        
        train_loader = train_utils.convert_to_tensor(Xtrain, Ytrain, batch_size=batch_size)
        val_loader = train_utils.convert_to_tensor(eeg_val, label_val, batch_size=batch_size)

        model, optimizer = model_fine_tuning_params(model_file, electrode_scale=electrode_scaling, 
                                                    sample_scale=sample_scale, dense=dense)

        trained_model = train_utils.model_training(model, train_loader, val_loader, Tuning=True)
        
        model_name = f'FineTunedEEGScale_S{sub:02d}.pth'
        out_dir = os.path.join(constants.mdl_dir, model_name)
        torch.save(model.state_dict(), out_dir)  # Save best model

        test_loader = train_utils.convert_to_tensor(Xte, Yte, batch_size=Yte.size)

        test_loss, test_acc, _ = train_utils.model_evaluation(trained_model, test_loader)
       
        print(f'Subject: {sub:02d}, Test Accuracy: {test_acc:.2f}%')
        online_perf[f'Sub{sub:02d}'] = test_acc

    print(f'Average Accuracy: {np.mean(list(online_perf.values()))}')
    print(list(online_perf.values()))

    return online_perf




if __name__=='__main__':
    # trained_model = model_using_calibration_data(verbose=True)
    print('Inferncing Only)')
    acc = evaluation_on_online_data('baseElectrodeScaler')

    print('Subject Specific Performance')
    online_acc = subject_specific_fine_tuning(electrode_scaling=True, sample_scale=True, dense=True)



    


