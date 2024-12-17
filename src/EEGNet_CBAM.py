import numpy as np
import pandas as pd
import os
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import LeaveOneOut

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

import direction_utils as utils
import direction_learning_utils as train_utils
import constants

SEED=constants.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) 
np.random.seed(SEED)
random.seed(SEED)

def model_using_calibration_data(verbose=False):
    # torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = constants.data_dir
    # print(data_dir)

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
    # torch.manual_seed(0)
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
    # torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mdl_dir = os.path.join(constants.mdl_dir, model_file)
    model = train_utils.EEGScaler(nb_classes=constants.classes, Chan=constants.chans, Samples=constants.samples).to(device)
    model.load_state_dict(torch.load(f'{mdl_dir}.pth'))

    if dense or electrode_scale or sample_scale:
        for param in model.parameters():
            param.requires_grad = False

        if electrode_scale:
            # print(f'Electrode Scaling are fine-tuned')
            for param in model.electrode_scale.parameters():
                param.requires_grad = True  # Unfreeze  layer
        
        if sample_scale:
            # print(f'Temporal Sample Scales are fine-tuned')
            for param in model.sample_scale.parameters():
                param.requires_grad = True  # Unfreeze  layer
        
        if dense:
            # print(f'Dense Layer is fine-tuned')
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


def subject_independent_speed_decoding(verbose=False):
    # torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')
    train_ratio = constants.train_ratio

    subids = list(range(1, 15))
    loo = LeaveOneOut()

    online_perf = dict()
    for train_idx, test_idx in loo.split(subids):
        train_subids = [subids[i] for i in train_idx]
        test_subids = [subids[i] for i in test_idx]        

        Xtr, Ytr, Xte, Yte = utils.speed_dataset(train_subids, test_subids, base_path=data_dir)
        eeg_train, eeg_val, label_train, label_val = train_test_split(Xtr, Ytr, 
                                                                train_size=train_ratio, random_state=42, shuffle=True)
        
        train_loader = train_utils.convert_to_tensor(eeg_train, label_train, train_flag=True)
        val_loader = train_utils.convert_to_tensor(eeg_val, label_val, train_flag=False)

        model = train_utils.EEGScaler(nb_classes=2, Chan=33, Samples=2000).to(device)
        trained_model = train_utils.model_training(model, train_loader, val_loader, verbose=verbose)

        test_loader = train_utils.convert_to_tensor(Xte, Yte, batch_size=Yte.size, train_flag=False)
        test_loss, test_acc, _ = train_utils.model_evaluation(trained_model, test_loader)
       
        print(f'Subject: {test_subids[0]:02d}, Test Accuracy: {test_acc:.2f}%')
        online_perf[f'Sub{test_subids[0]:02d}'] = test_acc
    
    print(f'Average Accuracy: {np.mean(list(online_perf.values()))}')
    print(list(online_perf.values()))

    return online_perf


# Subject Specific Model
def subject_specific_speed_decoding(verbose=False):
    # torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')
    train_ratio = constants.train_ratio

    subwise_acc = dict()
    for sub in range(1, 15):
        # Load the subjects' dataset
        rel_path = f'data/S{sub:02d}_midata.mat'
        mat_data = scipy.io.loadmat(os.path.join(data_dir, rel_path))
        X = mat_data['Xtrain']
        Y = mat_data['Ytrain']

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold=1
        fold_accuracies = []

        for train_idx, test_idx in kf.split(X):
            # Split into training and test sets
            Xtr, Xte = X[train_idx], X[test_idx]
            Ytr, Yte = Y[train_idx], Y[test_idx]


            eeg_train, eeg_val, label_train, label_val = train_test_split(Xtr, Ytr, 
                                                                train_size=train_ratio, random_state=42, shuffle=True)
        
            train_loader = train_utils.convert_to_tensor(eeg_train, label_train, train_flag=True)
            val_loader = train_utils.convert_to_tensor(eeg_val, label_val, train_flag=False)

            model = train_utils.EEGScaler(nb_classes=2, Chan=33, Samples=2000).to(device)
            trained_model = train_utils.model_training(model, train_loader, val_loader, verbose=verbose)

            test_loader = train_utils.convert_to_tensor(Xte, Yte, batch_size=Yte.size, train_flag=False)
            test_loss, test_acc, _ = train_utils.model_evaluation(trained_model, test_loader)

            fold_accuracies.append(test_acc)
            print(f"Fold {fold}: Accuracy = {test_acc:.4f}")
            fold += 1
       
        # Calculate and print average accuracy for the subject
        avg_accuracy = np.mean(fold_accuracies)
        subwise_acc[f'Sub{sub:2d}'] = avg_accuracy
        print(f"Subject {sub}: Average 5-Fold Accuracy = {avg_accuracy:.4f}")
    
    print(f'Average Accuracy: {np.mean(list(subwise_acc.values()))}')
    return 


def subjectwise_finetuned_speed_decoding(electrode_scaling=True, sample_scale=True, dense=True, verbose=False):
    # Train a subject-independent model 
    # torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = constants.data_dir #os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')
    train_ratio = constants.train_ratio

    subids = list(range(1, 15))
    loo = LeaveOneOut()

    subwise_acc = dict()
    for train_subs, test_subs in loo.split(subids):
        train_subids = [subids[i] for i in train_subs]
        test_subids = [subids[i] for i in test_subs] 

        train_data, train_label, test_data, test_label = utils.speed_dataset(train_subids, test_subids, base_path=data_dir)
        eeg_train, eeg_val, label_train, label_val = train_test_split(train_data, train_label, 
                                                                train_size=train_ratio, random_state=42, shuffle=True)
        
        train_loader = train_utils.convert_to_tensor(eeg_train, label_train, train_flag=False)
        val_loader = train_utils.convert_to_tensor(eeg_val, label_val, train_flag=False)
        # save the data as 
        # torch.save({'train_loader': train_loader})

        model = train_utils.EEGScaler(nb_classes=constants.classes, Chan=constants.chans, Samples=constants.samples).to(device)
        trained_model = train_utils.model_training(model, train_loader, val_loader, verbose=verbose)

        # Fine tuned the scaling layers only to report the performance. 
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold=1
        fold_accuracies = []

        for train_idx, test_idx in kf.split(test_data):
            # Split into training and test sets
            Xtr, Xte = test_data[train_idx], test_data[test_idx]
            Ytr, Yte = test_label[train_idx], test_label[test_idx]

            Xtrain, Xval, Ytrain, Yval = train_test_split(Xtr, Ytr, train_size=train_ratio, random_state=42, shuffle=True)
            
            train_loader = train_utils.convert_to_tensor(Xtrain, Ytrain, batch_size=constants.batch_size, train_flag=False)
            val_loader = train_utils.convert_to_tensor(Xval, Yval, batch_size=constants.batch_size, train_flag=False)

            finetuned_mdl, optimizer = model_fine_tuning_params('trained_model_checkpoint', electrode_scale=electrode_scaling, 
                                                        sample_scale=sample_scale, dense=dense)

            finetuned_mdl = train_utils.model_training(finetuned_mdl, train_loader, val_loader, Tuning=True, verbose=verbose)
            test_loader = train_utils.convert_to_tensor(Xte, Yte, batch_size=Yte.size, train_flag=False)

            test_loss, test_acc, _ = train_utils.model_evaluation(finetuned_mdl, test_loader)

            fold_accuracies.append(test_acc)
            print(f"Fold {fold}: Accuracy = {test_acc:.4f}")
            fold += 1
       
        # Calculate and print average accuracy for the subject
        avg_accuracy = np.mean(fold_accuracies)
        subwise_acc[f'Sub{test_subs[0]:2d}'] = avg_accuracy
        print(f"Subject {test_subs[0]}: Average 5-Fold Accuracy = {avg_accuracy:.4f}")
    
    print(f'Average Accuracy: {np.mean(list(subwise_acc.values()))}')
    return 


def subjectwise_finetuned_speed_decoding_v2(electrode_scaling=True, sample_scale=True, dense=True, verbose=False):
    # Train a subject-independent model 
    # torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = constants.data_dir #os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')
    train_ratio = constants.train_ratio

    subids = list(range(1, 15))
    loo = LeaveOneOut()

    subwise_acc = dict()
    for train_subs, test_subs in loo.split(subids):

        train_subids = [subids[i] for i in train_subs]
        test_subids = [subids[i] for i in test_subs] 
        # if test_subids[0]!=10:
        #     print(f'Test Subject: {test_subids[0]}')
        #     continue

        filepath = os.path.join(constants.data_dir, 'data', f'augmented_data_testsub_S{test_subids[0]:02d}.pt')
        augmented_data = torch.load(filepath)

        train_X_tensor = augmented_data['Xtr']
        train_Y_tensor = augmented_data['Ytr']
        
        val_X_tensor = augmented_data['Xval']
        val_Y_tensor = augmented_data['Yval']

        train_loader = DataLoader(dataset=TensorDataset(train_X_tensor, train_Y_tensor),
                                  batch_size=constants.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=TensorDataset(val_X_tensor, val_Y_tensor),
                            batch_size=constants.batch_size,
                            shuffle=False)

        model = train_utils.EEGScaler(nb_classes=constants.classes, Chan=constants.chans, Samples=constants.samples).to(device)
        trained_model = train_utils.model_training(model, train_loader, val_loader, verbose=verbose)

        fold_accuracies = []
        for fold in range(1, 6):
            filepath = os.path.join(constants.data_dir, 'data',  f'augmented_data_fold{fold}_S{test_subids[0]:02d}.pt')
            # print(f'  ...{filepath}')

            augmented_torch_data = torch.load(filepath)

            train_X_tensor = augmented_torch_data['Xtr']
            train_Y_tensor = augmented_torch_data['Ytr']
            
            val_X_tensor = augmented_torch_data['Xval']
            val_Y_tensor = augmented_torch_data['Yval']

            test_X_tensor = augmented_torch_data['Xte']
            test_Y_tensor = augmented_torch_data['Yte']

            train_loader = DataLoader(dataset=TensorDataset(train_X_tensor, train_Y_tensor),
                                    batch_size=constants.batch_size,
                                    shuffle=True)
            val_loader = DataLoader(dataset=TensorDataset(val_X_tensor, val_Y_tensor),
                                batch_size=constants.batch_size,
                                shuffle=False)
            test_loader = DataLoader(dataset=TensorDataset(test_X_tensor, test_Y_tensor),
                                     batch_size=constants.batch_size,
                                     shuffle=False)

            finetuned_mdl, optimizer = model_fine_tuning_params('trained_model_checkpoint', electrode_scale=electrode_scaling, 
                                                        sample_scale=sample_scale, dense=dense)

            finetuned_mdl = train_utils.model_training(finetuned_mdl, train_loader, val_loader, Tuning=True, verbose=verbose)

            test_loss, test_acc, _ = train_utils.model_evaluation(finetuned_mdl, test_loader)

            fold_accuracies.append(test_acc)
            print(f"Fold {fold}: Accuracy = {test_acc:.4f}")
            fold += 1
       
        # Calculate and print average accuracy for the subject
        avg_accuracy = np.mean(fold_accuracies)
        subwise_acc[f'Sub{test_subs[0]:2d}'] = avg_accuracy
        print(f"Subject {test_subs[0]}: Average 5-Fold Accuracy = {avg_accuracy:.4f}")
    
    print(f'Average Accuracy: {np.mean(list(subwise_acc.values()))}')
    return 


if __name__=='__main__':
    # trained_model = model_using_calibration_data(verbose=True)
    # print('Inferncing Only)')
    # acc = evaluation_on_online_data('baseElectrodeScaler')

    # print('Subject Specific Performance')
    # online_acc = subject_specific_fine_tuning(electrode_scaling=True, sample_scale=True, dense=True)

    # speed_acc = mi_speed_decoding(verbose=True)
    # subject_specific_speed_decoding(verbose=False)


    # subjectwise_finetuned_speed_decoding(electrode_scaling=True, sample_scale=True, dense=True, verbose=False)
    subjectwise_finetuned_speed_decoding_v2(electrode_scaling=False, sample_scale=True, dense=True, verbose=False)