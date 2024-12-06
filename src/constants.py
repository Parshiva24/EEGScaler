import os

batch_size=16
fs = 500
train_ratio = 0.9
train_sub_indx = 7 # Maximum, 0-7 are training subjects

learning_rate=1e-4
num_epochs = 1000
patience = 30  # Number of epochs to wait before stopping if no improvement
min_delta = 1e-3 # Minimum change to qualify as improvement

data_dir = os.path.join(os.getcwd(), 'SIT2024', 'MIDecoding_SENet')
mdl_dir = os.path.join(os.getcwd(), 'SIT2024', 'MI_TimeAttention')