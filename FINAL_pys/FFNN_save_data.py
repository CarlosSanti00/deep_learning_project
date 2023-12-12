import numpy as np
import seaborn as sns
import pandas as pd
import math
import torch
sns.set_style("whitegrid")
from torch import nn, Tensor, optim
from torch.nn.functional import softplus
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import h5py

# Feed-forward Neural network architecture
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout_rate=0):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    # Forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Class for reading GTEX dataset (including tissues labels for stratification of the dataset in CV folds)
class GtexExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, include: str = "", exclude: str = "", load_in_mem: bool = False, col_names: list = None):
        f_gtex_gene = h5py.File(data_dir + 'gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_isoform = h5py.File(data_dir + 'gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_gene = f_gtex_gene['expressions']
        
        # Filter columns in f_gtex_isoform based on col_names if provided
        if col_names:
            self.dset_isoform = f_gtex_isoform['expressions'][:, [col_index for col_index, col_name in enumerate(f_gtex_isoform['col_names'][:]) if col_name.decode() in col_names]]
        else:
            self.dset_isoform = f_gtex_isoform['expressions']

        self.tissue_labels = f_gtex_gene['tissue'][:]

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0] == len(self.tissue_labels))

        if load_in_mem:
            self.dset_gene = np.array(self.dset_gene)
            self.dset_isoform = np.array(self.dset_isoform)

        self.idxs = None

        if include and exclude:
            raise ValueError("You can only give either the 'include' or the 'exclude' argument.")

        if include:
            matches = [bool(re.search(include, s.decode(), re.IGNORECASE)) for s in self.tissue_labels]
            self.idxs = np.where(matches)[0]

        elif exclude:
            matches = [not(bool(re.search(exclude, s.decode(), re.IGNORECASE))) for s in self.tissue_labels]
            self.idxs = np.where(matches)[0]

    def __len__(self):
        if self.idxs is None:
            return self.dset_gene.shape[0]
        else:
            return len(self.idxs)

    def __getitem__(self, idx):
        if self.idxs is None:
            return torch.Tensor(self.dset_gene[idx]), torch.Tensor(self.dset_isoform[idx]), self.tissue_labels[idx]
        else:
            return torch.Tensor(self.dset_gene[self.idxs[idx]]), torch.Tensor(self.dset_isoform[self.idxs[idx]]), self.tissue_labels[self.idxs[idx]]


# Load the dataset (predicting all isoform ids)
gtex_all_data = GtexExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")

# Model parameters
input_size = len(gtex_all_data[0][0])
n_hidden_1 = 512
n_hidden_2 = 16
output_size = 156958

# Instantiate the model with dropout
model = RegressionModel(input_size, n_hidden_1, n_hidden_2, output_size, dropout_rate=0)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Saving the all possible tissues for stratification
tissue_labels = np.array([tissue for _, _, tissue in gtex_all_data])

# Model training and stratification parameters
num_folds = 5
batch_size = 64
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store individual validation raw errors for the last model of each fold 
all_raw_errors = []

for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(np.zeros(len(tissue_labels)), tissue_labels)):

    # Random sampling of the stratified partition
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Load the dataloader according to the sampler
    gtex_train_dataloader = DataLoader(gtex_all_data, batch_size=batch_size, sampler=train_sampler)
    gtex_test_dataloader = DataLoader(gtex_all_data, batch_size=batch_size, sampler=test_sampler)

    print(f"\nStarting Fold {fold + 1}:")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Train the model
    n_epochs = 5
    for epoch in range(1, n_epochs+1, 1):
        
        # MSE total train vector definition for each epoch
        mse_train = 0
        for X, y, _ in tqdm(gtex_train_dataloader):

            # Forward pass
            outputs = model(X.float())  
            loss = criterion(outputs, y.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metric calculation (training)
            mse_train_batch = loss.item()
            mse_train += mse_train_batch

        # Calculate the mean MSE for the total number of batches in your training set
        mean_mse_train = mse_train / len(gtex_train_dataloader)

        # Validation calculation
        mse_valid = 0
        baseline_mse_valid = 0
        model.eval()
        with torch.no_grad():
            print(f'Number of batches of valid:{len(gtex_test_dataloader)}')
            for X_val, y_val, _ in (gtex_test_dataloader):

                outputs_val = model(X_val.float())
                valid_loss = criterion(outputs_val, y_val.float())
                mse_valid_batch = valid_loss.item()
                mse_valid += mse_valid_batch

                if epoch == n_epochs:
                    
                    # Baseline predictions
                    baseline_predictions = torch.mean(y_val.float(), dim=0, keepdim=True).repeat(len(y_val), 1)
                    print(f'Baseline pred shape: {baseline_predictions.shape}')
                    print(f'Y_val shape: {y_val.shape}')
                    baseline_loss = criterion(baseline_predictions, y_val.float())
                    baseline_mse_valid_batch = baseline_loss.item()
                    baseline_mse_valid += baseline_mse_valid_batch
                
                # # Store individual absolute errors for the current fold and last epoch
                # if epoch == n_epochs:
                #     print('Comparison thing:')
                #     print(y_val.shape)
                #     print(outputs_val.shape)
                #     raw_errors = (outputs_val - y_val.float()).numpy()
                #     print(raw_errors.shape)
                #     row_means = np.mean(raw_errors, axis=1)
                #     print(row_means)
                #     # all_raw_errors.extend(raw_errors)

            # Mean mse value according all batches
            mean_mse_valid = mse_valid / len(gtex_test_dataloader)
            baseline_mean_mse_valid = baseline_mse_valid / len(gtex_test_dataloader)

        # Print metric results
        if (epoch) >= 10:
            if epoch % 10 == 0:
                tqdm.write(f'\nEpoch {epoch}: train loss (last batch), mean MSE:\t{mse_train_batch:.4f},\t{mean_mse_train:.5f}')
                tqdm.write(f'Epoch {epoch}: valid loss (last batch), mean MSE:\t{mse_valid_batch:.4f},\t{mean_mse_valid:.5f}')
                tqdm.write(f'Epoch {epoch}: valid baseline mean MSE:\t{baseline_mean_mse_valid:.5f}')


# np.savetxt('valid_errors.txt', np.array(all_raw_errors))

# Histogram plot
# plt.hist(all_raw_errors, bins=10, edgecolor='black')
# plt.title('Validation Raw Error Histogram')
# plt.xlabel('Error Value')
# plt.ylabel('Frequency')
# plt.savefig('validation_me_histogram_all.png')
# plt.show()

# # Create a density plot
# sns.distplot(np.array(all_raw_errors), kind='kde')
# # Set plot labels and title
# plt.title('Validation Raw Error Density Plot')
# plt.xlabel('Error Value')
# plt.ylabel('Density')
# # Save or display the plot
# plt.savefig('validation_me_density_plot.png')
# plt.show()

# # Create a KDE plot with shade
# sns.kdeplot(np.array(all_raw_errors), fill=True)
# # Set plot labels and title
# plt.title('Kernel Density Estimation (KDE) plot with Shade')
# plt.xlabel('Validation error value')
# plt.ylabel('Density')
# # Save or display the plot
# plt.savefig('kde_plot_with_shade.png')
# plt.show()