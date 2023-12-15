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
from scipy.stats import ttest_rel

# Feed-forward Neural network architecture
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Class for reading GTEX dataset (including tissues labels for stratification of the dataset in CV folds)
class GtexExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, include: str = "", exclude: str = "", load_in_mem: bool = False, col_names: list = None, VAE: bool = False):
        f_gtex_gene = h5py.File(data_dir + 'gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_latent = h5py.File('../VAE_settings/latent_features.h5', mode='r')
        f_gtex_isoform = h5py.File(data_dir + 'gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        # self.dset_gene = f_gtex_gene['expressions']
        if (VAE == True):
            self.dset_gene = f_gtex_latent['latent_features']
        else:
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
gtex_all_data_og = GtexExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/", VAE = False)
gtex_all_data = GtexExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/", VAE = True)

# Model parameters
input_size = len(gtex_all_data[0][0])
input_size_og = len(gtex_all_data_og[0][0])
n_hidden = 32
output_size = 156958

# Saving the all possible tissues for stratification
tissue_labels = np.array([tissue for _, _, tissue in gtex_all_data])
tissue_labels_og = np.array([tissue for _, _, tissue in gtex_all_data_og])

# Model training and stratification parameters
num_folds = 5
batch_size = 64
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store individual validation raw errors for the last model of each fold 
all_raw_errors = []

for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(np.zeros(len(tissue_labels_og)), tissue_labels_og)):

    # Random sampling of the stratified partition
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Load the dataloader according to the sampler
    # For the VAE
    gtex_train_dataloader = DataLoader(gtex_all_data, batch_size=batch_size, sampler=train_sampler)
    gtex_test_dataloader = DataLoader(gtex_all_data, batch_size=batch_size, sampler=test_sampler)
    # For the original
    gtex_train_dataloader_og = DataLoader(gtex_all_data_og, batch_size=batch_size, sampler=train_sampler)
    gtex_test_dataloader_og = DataLoader(gtex_all_data_og, batch_size=batch_size, sampler=test_sampler)

    print(f"\nStarting Fold {fold + 1}:")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Instantiate the model with dropout
    model = RegressionModel(input_size, n_hidden, output_size, dropout_rate=0)
    model_og = RegressionModel(input_size_og, n_hidden, output_size, dropout_rate=0)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_og = optim.Adam(model_og.parameters(), lr=1e-3)

    # To store the MSE values of the last epoch
    vae_mse_epochs = []
    non_vae_mse_epochs = []
    baseline_mse_epochs = []

    # Train the model
    n_epochs = 100
    for epoch in range(1, n_epochs+1, 1):
        
        # MSE total train vector definition for each epoch
        mse_train = 0
        mse_train_og = 0
        # TRAIN MODEL WITH THE VAE DATA
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

            # Baseline
            baseline_mean = torch.mean(y.float(), dim=0, keepdim=True)
        
        # TRAIN MODEL WITH THE ORIGINAL DATA
        for X_og, y_og, _ in tqdm(gtex_train_dataloader_og):

            # Forward pass
            outputs_og = model_og(X_og.float())  
            loss_og = criterion(outputs_og, y_og.float())

            # Backward pass and optimization
            optimizer_og.zero_grad()
            loss_og.backward()
            optimizer_og.step()

            # Metric calculation (training)
            mse_train_batch_og = loss_og.item()
            mse_train_og += mse_train_batch_og

        # Calculate the mean MSE for the total number of batches in your training set
        mean_mse_train = mse_train / len(gtex_train_dataloader)
        mean_mse_train_og = mse_train_og / len(gtex_train_dataloader_og)

        # Validation calculation
        mse_valid = 0
        mse_valid_og = 0
        baseline_mse_valid = 0
        manual_mse_baseline = 0
        model.eval()
        with torch.no_grad():
            # print(f'Number of batches of valid:{len(gtex_test_dataloader)}') # 55
            for X_val, y_val, _ in (gtex_test_dataloader):

                # FOR THE VAE MODEL
                outputs_val = model(X_val.float())
                valid_loss = criterion(outputs_val, y_val.float())
                mse_valid_batch = valid_loss.item()
                mse_valid += mse_valid_batch
                    
                # Baseline predictions
                baseline_predictions = baseline_mean.repeat(len(y_val), 1)
                baseline_loss = criterion(baseline_predictions, y_val.float())
                baseline_mse_valid_batch = baseline_loss.item()
                baseline_mse_valid += baseline_mse_valid_batch
                    
                if (epoch) == 100:
                    vae_mse_epochs.append(mse_valid_batch)
                    baseline_mse_epochs.append(baseline_mse_valid_batch)

            for X_val, y_val, _ in (gtex_test_dataloader_og):
                # FOR THE NON-VAE MODEL
                outputs_val_og = model_og(X_val.float())
                valid_loss_og = criterion(outputs_val_og, y_val.float())
                mse_valid_batch_og = valid_loss_og.item()
                mse_valid_og += mse_valid_batch_og

                if (epoch) == 100:
                    non_vae_mse_epochs.append(mse_valid_batch_og)


            # Mean mse value according all batches
            mean_mse_valid = mse_valid / len(gtex_test_dataloader)
            mean_mse_valid_og = mse_valid_og / len(gtex_test_dataloader_og)
            baseline_mean_mse_valid = baseline_mse_valid / len(gtex_test_dataloader)

            
        # Print metric results
        if (epoch) >= 5:
            if epoch % 5 == 0:
                tqdm.write(f'\nEpoch {epoch} - VAE model: train loss (last batch), mean MSE:\t{mse_train_batch:.4f},\t{mean_mse_train:.5f}')
                tqdm.write(f'Epoch {epoch} - VAE model: valid loss (last batch), mean MSE:\t{mse_valid_batch:.4f},\t{mean_mse_valid:.5f}')
                tqdm.write(f'\nEpoch {epoch} - NON-VAE model: train loss (last batch), mean MSE:\t{mse_train_batch_og:.4f},\t{mean_mse_train_og:.5f}')
                tqdm.write(f'Epoch {epoch}: - NON-VAE model: valid loss (last batch), mean MSE:\t{mse_valid_batch_og:.4f},\t{mean_mse_valid_og:.5f}')
                tqdm.write(f'\nEpoch {epoch}: valid baseline mean MSE:\t{baseline_mean_mse_valid:.5f}')

    # TESTING
    print(len(vae_mse_epochs))
    print(len(non_vae_mse_epochs))
    p_value_vae_vs_non_vae = ttest_rel(vae_mse_epochs, non_vae_mse_epochs).pvalue
    p_value_vae_vs_baseline = ttest_rel(vae_mse_epochs, baseline_mse_epochs).pvalue
    p_value_non_vae_vs_baseline = ttest_rel(non_vae_mse_epochs, baseline_mse_epochs).pvalue

    print(f"\nFold {fold + 1} Results:")
    print(f"VAE MSE: {mean_mse_valid}")
    print(f"Non-VAE MSE: {mean_mse_valid_og}")
    print(f"Baseline MSE: {baseline_mean_mse_valid}")
    print(f"VAE vs. Non-VAE p-value: {p_value_vae_vs_non_vae}")
    print(f"VAE vs. Baseline p-value: {p_value_vae_vs_baseline}")
    print(f"Non-VAE vs. Baseline p-value: {p_value_non_vae_vs_baseline}")
