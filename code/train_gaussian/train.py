## Training function
## Load data, initialize model, start training, and store relevant data during training
## Store parameters for each epoch, the best test parameters, and validation parameters
## Call other functions to plot train and validation curves, etc.
## Xie Yuxuan 2024-08-22 UTF-8
import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import cnn_trans 
import plot
import shutil
from ptflops import get_model_complexity_info
from toolbox import batch_extract_upper_triangle
from tqdm import tqdm

# Name of the experiment, used to create a folder in weight_save_path to distinguish experiments with different settings
# Set the experiment name here before running the code each time
exp_name = 'exp-gaussian_grid01_1200_snap1000_2src'

# Set hyperparameters
batch_size = 64
epochs = 500

data_path = '/home/yxxie/Documents/cnn_transformer/training_data_gauss/data.npz'  # Path to training data
weight_path = '/home/yxxie/Documents/cnn_transformer/weights'  # Path to save model-related information for all experiments
save_path = os.path.join(weight_path, exp_name)  # Path to save information, including weights for each epoch, best weights, various images, training process logs, etc.

if sys.argv[0].endswith('train.py'):    # Execute the following logic only when this program is the main program. This is set so other programs can call save_path.
    # Create save path and check if it already exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Weight save directory '{save_path}' created successfully!")
        current_file_dir = os.path.dirname(os.path.abspath(__file__))   # Get the directory of the current Python file
        destination_folder = os.path.join(save_path, os.path.basename(current_file_dir))   # Construct the target folder path (save location)
        shutil.copytree(current_file_dir, destination_folder)  # Copy contents of the parent directory
        print(f"Training code successfully copied to the weight save directory: {destination_folder}")
    else:
        print(f"Weight save directory '{save_path}' already exists. The training program with this experiment name has been executed before.")
        user_input = input("If the previous training was for debugging, delete the directory and recreate it? Enter 'del' to delete: ")
        if user_input.lower() == 'del':
            shutil.rmtree(save_path)  # Delete the directory and its contents
            print(f"Old weight save directory '{save_path}' deleted!")
            os.makedirs(save_path)
            print(f"New weight save directory '{save_path}' created successfully!")
            current_file_dir = os.path.dirname(os.path.abspath(__file__))   # Get the directory of the current Python file
            destination_folder = os.path.join(save_path, os.path.basename(current_file_dir))   # Construct the target folder path (save location)
            shutil.copytree(current_file_dir, destination_folder)  # Copy contents of the parent directory
            print(f"Training code successfully copied to the weight save directory: {destination_folder}")
        else:
            print("Training operation canceled, the program will terminate.")
            sys.exit()
    

    # Create a dataset class for training
    # Contains over 4.3 million data entries
    class NumpyDataset(Dataset):
        def __init__(self, file_path):  # Load data and labels, store in two attributes
            data = np.load(file_path)
            self.data = data['data']
            self.label = data['label']
            data.close()
        
        def __len__(self):  # Return the total length of the data
            return len(self.data)
        
        def __getitem__(self, idx):  # Return data and labels for a specified index
            sample = self.data[idx]
            label = self.label[idx]
            # Convert to PyTorch tensor and adjust dimensions to fit CNN input
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = sample.permute(2, 0, 1)  # From [height, width, channels] to [channels, height, width]
            # sample = extract_upper_triangle(sample)  # Extract upper triangular elements, return shape (3, 120)
            # sample = sample.view(3, 1, 120)
            label = torch.tensor(label, dtype=torch.float32)
            return sample, label

    # Pass the file path list to create the dataset
    dataset = NumpyDataset(data_path)

    # Split into training and validation sets
    train_size = int(0.85 * len(dataset))  # 85% for training
    val_size = len(dataset) - train_size  # Remaining 15% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Get dataset length for computing the total loss per epoch
    train_data_num = len(train_dataset)
    val_data_num = len(val_dataset)

    # Initialize best loss values
    train_best_loss = float('inf')
    val_best_loss = float('inf')

    # Record the number of epochs the best loss has been maintained
    num_train = 0
    num_val = 0

    # Create the model, set loss function and optimizer
    model = cnn_trans()  # Create model instance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    flops, params = get_model_complexity_info(model, (3, 1, 120), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    criterion = nn.MSELoss()  # MSE loss for Gaussian-smoothed continuous labels
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with initial learning rate

    # Save information on various metrics over epochs for plotting
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    learning_rates = np.zeros(epochs)

    # Start training
    for epoch in tqdm(range(epochs)):

        #### Preparation ####
        train_loss = 0.0  # Training loss
        val_loss = 0.0  # Validation loss
        t1 = time.process_time()  # Record time

        # #### Record current learning rate ####
        # for param_group in optimizer.param_groups:
        #     learning_rates[epoch] = param_group['lr']

        #### Start training ####
        model.train()  # Set model to training mode
        for batch_data, batch_label in train_loader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)  # Move to GPU
            batch_data = batch_extract_upper_triangle(batch_data)  # Extract upper triangular elements, return shape (3, 136)
            batch_data = batch_data.view(batch_data.size(0), 3, 1, -1)
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(batch_data)  # Forward pass
            loss = criterion(outputs, batch_label)  # Compute loss
            loss.backward()  # Backward pass to update parameters
            optimizer.step()  # Adjust learning rate
            train_loss += loss.item() # Accumulate the total loss for this batch
        
        #### Start validation ####
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for batch_data, batch_label in val_loader:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
                batch_data = batch_extract_upper_triangle(batch_data)  # Extract upper triangular elements, return shape (3, 136)
                batch_data = batch_data.view(batch_data.size(0), 3, 1, -1)
                outputs = model(batch_data)  # Forward pass
                loss = criterion(outputs, batch_label)  # Compute loss
                val_loss += loss.item()    # Accumulate the total loss for this batch

        #### Compute and record the average loss and RMSE for this epoch ####
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        #### Save weights for this epoch ####
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}_weights.pth'))

        #### Update and save best loss values ####
        num_train += 1
        if train_loss < train_best_loss:
            num_train = 0
            train_best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_train.pth'))
        
        num_val += 1
        if val_loss < val_best_loss:
            num_val = 0
            val_best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_val.pth'))

        #### Print and save training information ####
        log_message = "epoch:{}, train_loss:{}, val_loss:{}, train_best_loss:{}, num_train:{}, val_best_loss:{}, num_val:{}".format(
            epoch, train_loss, val_loss, train_best_loss, num_train, val_best_loss, num_val,
        )
        elapsed_time = time.process_time() - t1

        # Open file and write information
        with open(os.path.join(save_path,'train_info.txt'), 'a') as f:
            f.write(log_message + '\n')
            f.write("Time for this epoch: {:.2f} seconds\n".format(elapsed_time))

        # Print information to the console
        print(log_message)
        print("Time for this epoch: {:.2f} seconds\n".format(elapsed_time))

        # Plot training and validation loss curves and learning rate curves
        plot.plot_train_and_val_loss(train_losses, val_losses, save_path)
        # plot.plot_lr(learning_rates, save_path)