# Module loads 

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
import h5py


import math
import torch
import IsoDatasets
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, LogNormal
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from collections import defaultdict
from torchvision.transforms import ToTensor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.optim as optim



class Autoencoder(nn.Module):

    def __init__(self, input_shape: torch.Size, latent_features: int) -> None:
        super(Autoencoder, self).__init__()

        self.input_shape = input_shape
        self.observation_features = np.prod(input_shape)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.observation_features),
        )

    def forward(self, x) -> Dict[str, Any]:
        """compute the encoder and decoder outputs"""
        # Convert the input to a PyTorch tensor
        x = torch.Tensor(x)
        # flatten the input
        x = x.view(x.size(0), -1)

        # encode x
        encoded = self.encoder(x)

        # decode the encoded representation
        decoded = self.decoder(encoded)

        return {'decoded': decoded, 'encoded': encoded}


# Load the data

train_batch_size = 64
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
archs4_train_dataloader = DataLoader(archs4_train, batch_size=train_batch_size, shuffle=True)
print("Archs4 training set size:", len(archs4_train))

# Define the test sets (gtex_gene_expression)
eval_batch_size = 64
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", include='brain')
gtex_test_dataloader = DataLoader(gtex_test, batch_size=eval_batch_size, shuffle=True)
print("Gtex test set size:", len(gtex_test))


batch_size = 64


# Create an instance of the Autoencoder
latent_features = 256
autoencoder = Autoencoder(archs4_train[0].shape, latent_features)

# The Adam optimizer works well for autoencoders
autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)


# Initialize lists to store MSE values
all_training_mse = []
all_validation_mse = []

# Training
num_epochs = 5
epoch = 0

while epoch < num_epochs:

    # Set to training mode
    autoencoder.train()

    for i, x in enumerate(archs4_train_dataloader):

        # Check the condition to continue with a new epoch
        if i > 100:
            break

        # Forward pass through the autoencoder
        autoencoder_outputs = autoencoder(x)

        # Compute MSE during training
        mse_train = nn.MSELoss()(autoencoder_outputs['decoded'], x)
        all_training_mse.append(mse_train.item())

        # Backward pass and optimization
        autoencoder_optimizer.zero_grad()
        mse_train.backward()
        autoencoder_optimizer.step()

    # Set to evaluation mode
    autoencoder.eval()

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():

        for i, x in enumerate(gtex_test_dataloader):

            # Check the condition to continue with a new epoch
            if i > 100:
                break

            # Forward pass through the autoencoder
            autoencoder_outputs = autoencoder(x)

            # Compute MSE during validation
            mse_val = nn.MSELoss()(autoencoder_outputs['decoded'], x)
            all_validation_mse.append(mse_val.item())

    # Print the MSE values for training and validation
    print(f'Epoch {epoch}/{num_epochs} => Training MSE: {np.mean(all_training_mse):.4f}, Validation MSE: {np.mean(all_validation_mse):.4f}')

    epoch += 1

# Plot the training and validation MSE values across epochs
fig, ax = plt.subplots()
ax.set_title('Mean Squared Error (MSE) across Epochs')
ax.plot(all_training_mse, label='Training MSE')
ax.plot(all_validation_mse, label='Validation MSE')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
plt.show()