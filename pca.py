# Module loads
from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
from plotting import make_vae_plots


import math
import torch
import IsoDatasets
from torch import nn, Tensor
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from collections import defaultdict

# Define the train sets
train_batch_size = 64
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
archs4_train_dataloader = DataLoader(archs4_train, batch_size=train_batch_size, shuffle=True)
print("Archs4 training set size:", len(archs4_train))

# Define the test sets (gtex_gene_expression)
eval_batch_size = 64
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", include='brain')
gtex_test_dataloader = DataLoader(gtex_test, batch_size=eval_batch_size, shuffle=True)
print("Gtex test set size:", len(gtex_test))

def apply_pca_on_dataloader(dataloader, n_components=2):
    # Extract gene expression data from batches
    all_data = []
    for batch in dataloader:
        gene_expression_data = batch.cpu().numpy()  # Assuming your data is in numpy format
        all_data.append(gene_expression_data)

    # Combine data from all batches
    combined_data = np.concatenate(all_data, axis=0)

    # Standardize the data
   # scaler = StandardScaler()
   # standardized_data = scaler.fit_transform(combined_data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(combined_data)

    return principal_components

# Apply PCA on the training DataLoader
archs4_pca = apply_pca_on_dataloader(archs4_train_dataloader)

# Apply PCA on the test DataLoader
gtex_pca = apply_pca_on_dataloader(gtex_test_dataloader)

# Visualize Results
plt.figure(figsize=(10, 6))
plt.scatter(archs4_pca[:, 0], archs4_pca[:, 1], label='Archs4 Training Data')
plt.scatter(gtex_pca[:, 0], gtex_pca[:, 1], label='Gtex Test Data')
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


plt.savefig('pca.png')
