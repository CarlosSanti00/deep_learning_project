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
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        print(self.observation_features)

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
      
        # flatten the input
        x = x.view(x.size(0), -1)

        # encode x
        encoded = self.encoder(x)

        # decode the encoded representation
        decoded = self.decoder(encoded)

        return {'decoded': decoded, 'encoded': encoded}



# Define a custom transformation for dynamic normalization
class DynamicNormalizeTransform:
    def __init__(self):
        self.min_values = None
        self.max_values = None

    def __call__(self, samples):

        # Convert the list of arrays to a list of PyTorch tensors
        samples = [torch.from_numpy(x[0]) if isinstance(x, tuple) else torch.from_numpy(x) for x in samples]

        # Convert the list of tensors to a stacked tensor
        x = torch.stack(samples, dim=0)

        # Calculate min and max dynamically
        if self.min_values is None or self.max_values is None:
            self.min_values = x.min(dim=1, keepdim=True).values
            self.max_values = x.max(dim=1, keepdim=True).values

        # Apply normalization
        x = (x - self.min_values) / (self.max_values - self.min_values)

        return x




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





# Create an instance of the Autoencoder
latent_features = 256
autoencoder = Autoencoder(archs4_train[0].shape, latent_features)

# Maybe change this to SGD???
autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)



# Define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

# Initialize training loop
epoch = 0
num_epochs = 10


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f">> Using device: {device}")


# Move the model to the device
autoencoder = autoencoder.to(device)


# Initialize lists to store MSE values
all_training_mse = []
all_validation_mse = []

# Define the number of samples to print and save
num_samples = 5  
num_proteins = 10

while epoch < num_epochs:

    epoch += 1
    training_epoch_data = defaultdict(list)
    autoencoder.train()

    # Shuffle the data loader for each epoch
    archs4_train_dataloader = DataLoader(archs4_train, batch_size=train_batch_size, shuffle=True)

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here

    # try:
    for i, x in enumerate(archs4_train_dataloader):

        # Check the condition to continue with a new epoch
        if i > 100:
            break

        x = x.to(device)
        pseudocount = 1e-8
        x = x + pseudocount

        autoencoder_outputs = autoencoder(torch.Tensor(x))
        decoded = autoencoder_outputs['decoded']

        # Forward pass through the autoencoder
        autoencoder_outputs = autoencoder(torch.Tensor(x))
        decoded = autoencoder_outputs['decoded']

        # Compute MSE during training
        mse_train = nn.MSELoss()(decoded, torch.Tensor(x))
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
            x = x.to(device)
            pseudocount = 1e-8
            x = x + pseudocount
            autoencoder_outputs = autoencoder(torch.Tensor(x))

            # Compute MSE during validation
            mse_val = nn.MSELoss()(autoencoder_outputs['decoded'], torch.Tensor(x))
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