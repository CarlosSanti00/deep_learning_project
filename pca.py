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
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from collections import defaultdict

import h5py
from tqdm import tqdm



n_components = 12
eval_batch_size = 64 
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/") 
gtex_test_dataloader = DataLoader(gtex_test, batch_size=eval_batch_size, shuffle=True) 
ipca = IncrementalPCA(n_components=n_components)
print(type(gtex_test_dataloader)) 
# Print the first 5 batches of the DataLoader
for batch_index, (inputs, labels) in enumerate(gtex_test_dataloader):
    if batch_index < 5:
        print(f"Batch {batch_index + 1}:")
        print("Inputs:")
        print(inputs)
        print("Labels:")
        print(labels)
       
    else:
        break
# Accumulate data for PCA
all_data = []
for batch_index, (inputs, _) in enumerate(gtex_test_dataloader):
    all_data.append(inputs.numpy())

# Concatenate data and fit PCA
all_data = np.concatenate(all_data, axis=0)
ipca.fit(all_data)

# Transform data with IPCA
transformed_data = ipca.transform(all_data)

# Plot the first two principal components
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
plt.title('Principal Components of Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
plt.savefig("PCA_gtex.png")
#all_principal_components_gtex = []
# Iterate over batches in the dataloader and update the principal components
#for X in tqdm(gtex_test_dataloader):


    # Convert PyTorch tensors to NumPy arrays
 #   X_numpy = [tensor.numpy() for tensor in X]

    # Find the maximum length of sequences
  #  max_length = max(len(seq) for seq in X_numpy)

    # Pad sequences to the maximum length
   # X_padded = [np.concatenate([seq, np.zeros(max_length - len(seq))]) for seq in X_numpy]

    # Convert to NumPy array
    #X_array = np.array(X_padded)
     
    #ipca.partial_fit(X_array)
    #principal_components = ipca.transform(X.numpy())
    #all_principal_components_gtex.append(principal_components)
#all_principal_components_gtex = np.concatenate(all_principal_components_gtex, axis=0)
#print(all_principal_components_gtex)
# Plotting the first two principal components
#plt.scatter(all_principal_components_gtex[:, 0], all_principal_components_gtex[:, 1], alpha=0.5)
#plt.title('Principal Components of Gene Expression Data')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.show()
#plt.savefig("PCA_gtex.png")

#output_file_path = "principal_components_gtex.h5"
#with h5py.File(output_file_path, 'w') as hf:
 #   hf.create_dataset("principal_components_gtex", data=all_principal_components_gtex)

