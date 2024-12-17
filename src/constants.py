import os

classes = 2
chans = 33
samples = 2000
fs = 500
train_ratio = 0.9
train_sub_indx = 7 # Maximum, 0-7 are training subjects
seed=42


batch_size=16
learning_rate=1e-3
min_epochs = 20
best_val_acc = 70.00
num_epochs = 100
patience = 5  # Number of epochs to wait before stopping if no improvement
min_delta = 1e-4 # Minimum change to qualify as improvement
el_scaler_red_rate = 2
sample_scaler_red_rate = 4

data_dir = os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')
mdl_dir = os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')