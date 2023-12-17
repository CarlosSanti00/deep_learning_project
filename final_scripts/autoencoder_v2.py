# Module loads
from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
import h5py

import torch
from torch import nn
from torch.utils.data import DataLoader
from functools import reduce
from collections import defaultdict

# Your Autoencoder class definition
class AutoEncoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int) -> None:
        super(AutoEncoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
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

        # Calculate Mean Squared Error (MSE) as the reconstruction loss
        mse_loss = nn.MSELoss()(x_reconstructed, x.view(x.size(0), -1))

        return {'mse_loss': mse_loss}

# Module for training
class AutoencoderTraining:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.training_losses = []
        self.validation_losses = []

    def train_step(self, x: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        outputs = self.model(x)
        mse_loss = outputs['mse_loss']
        mse_loss.backward()
        self.optimizer.step()
        return mse_loss.item()

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                x_flat = x.view(x.size(0), -1)
                outputs = self.model(x_flat)
                total_loss += outputs['mse_loss'].item()
        average_loss = total_loss / len(dataloader)
        return average_loss

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for i, x in enumerate(train_dataloader):
                x = x.to(self.device)
                x_flat = x.view(x.size(0), -1)
                loss = self.train_step(x_flat)
                total_loss += loss

            average_loss = total_loss / len(train_dataloader)
            self.training_losses.append(average_loss)

            # Evaluate on the validation set
            val_loss = self.evaluate(val_dataloader)
            self.validation_losses.append(val_loss)

            print(f"Epoch [{epoch}/{num_epochs}] - Training MSE Loss: {average_loss:.4f} - Validation MSE Loss: {val_loss:.4f}")



# Your DataLoader setup
train_batch_size = 64
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/path/to/your/training/data/hdf5/")
archs4_train_dataloader = DataLoader(archs4_train, batch_size=train_batch_size, shuffle=True)

# Initialize the Autoencoder and DataLoader
latent_features = 256
autoencoder = AutoEncoder(input_shape=(your_input_size), latent_features=latent_features)
autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
autoencoder_trainer = AutoencoderTraining(model=autoencoder, optimizer=autoencoder_optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')

# Train the Autoencoder
num_epochs = 3
autoencoder_trainer.train(archs4_train_dataloader, your_validation_dataloader, num_epochs)

# Plot the MSE loss values across epochs and save as PNG
fig, ax = plt.subplots()
ax.set_title('Training Loss across Epochs')
ax.plot(autoencoder_trainer.training_losses, label='Training MSE Loss')
ax.plot(autoencoder_trainer.validation_losses, label='Validation MSE Loss')
ax.legend()
fig.savefig('mse_loss_plot.png')
plt.show()
