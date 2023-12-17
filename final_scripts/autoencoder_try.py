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
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from collections import defaultdict

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
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=latent_features),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.observation_features)
        )

    def forward(self, x) -> Dict[str, Any]:
        """Forward pass through the autoencoder"""

        # Flatten the input
        x = x.view(x.size(0), -1)

        # Encode
        z = self.encoder(x)

        # Decode
        x_reconstructed = self.decoder(z)

        return {'x_reconstructed': x_reconstructed, 'z': z}

# Module for training and evaluation
class AutoencoderTraining(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model: nn.Module, x: Tensor) -> Tuple[Tensor, Dict]:
        # Forward pass through the model
        outputs = model(x)

        # Unpack outputs
        x_reconstructed, z = [outputs[k] for k in ["x_reconstructed", "z"]]

        # Compute the mean squared error loss
        loss = nn.MSELoss()(x_reconstructed, x)

        # Prepare the output
        with torch.no_grad():
            diagnostics = {'mse_loss': loss.item()}

        return loss, diagnostics, outputs


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

# Initialization of the model, evaluator and optimizer

# VAE
latent_features = 256
print(f'Shape of the archs4 dataset (hd5): {archs4_train[0].shape}')
print(f'Shape of the gtex dataset (hd5): {gtex_test[0][0].shape}')




# Training configuration
latent_features = 256
autoencoder = Autoencoder(archs4_train[0].shape, latent_features)

autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

autoencoder_training = AutoencoderTraining()

# Training loop
num_epochs = 100

device = torch.device("cpu")
autoencoder = autoencoder.to(device)

all_training_losses = []

for epoch in range(1, num_epochs + 1):

    autoencoder.train()

    # Shuffle the data loader for each epoch
    archs4_train_dataloader = DataLoader(archs4_train, batch_size=train_batch_size, shuffle=True)

    for i, x in enumerate(archs4_train_dataloader):
        x = x.to(device)
        pseudocount = 1e-8
        x = x + pseudocount

        # Perform a forward pass through the model and compute the MSE loss
        loss, _, _ = autoencoder_training(autoencoder, x)

        autoencoder_optimizer.zero_grad()
        loss.backward()
        autoencoder_optimizer.step()

        # Accumulate the losses for each iteration
        all_training_losses.append(loss.item())

    # Print the training loss for every 5 epochs
    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - Training Loss: {loss.item()}")

# Save the trained autoencoder
autoencoder_path = "autoencoder.pth"
torch.save(autoencoder.state_dict(), autoencoder_path)
print(f"Autoencoder saved to {autoencoder_path}")

# Save the training losses to a file
np.savetxt("training_losses.txt", all_training_losses)

# Plot the training loss values across iterations and save as PNG
fig, ax = plt.subplots()
ax.set_title('Training Loss across Iterations')
ax.plot(all_training_losses, label='Training Loss')
ax.legend()
fig.savefig('loss_plot_Autoencoder.png')
plt.close(fig)
