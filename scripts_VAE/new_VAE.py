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
import torch.nn.functional as functional
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.distributions import LogNormal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from collections import defaultdict

# Inspiration from: 
# - https://github.com/NMinster/GeneExpressionVAE/blob/main/CVAE.py
# - https://mbernste.github.io/posts/vae/

######################################
## --- Variational Autoencoder ---  ##
######################################

# Degine the Variational Autoencoder Class

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=512, out_features=2*latent_features) # <- note the 2*latent_features
            # nn.Sigmoid()  # Sigmoid activation to the last layer
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=self.observation_features)
        )

        # Layer for outputting both the mean and the logarithm of the variance for the lateent distribution in the VAE

        self.enc_mu_logsig = nn.Linear(in_features = 512), out_features = 2*self.latent_features)

        # We proceed with the Xavier initializatio to set the initial weights, in this case we want
        # to set the weights of the linnear layers
        torch.nn.init.xavier_uniform_(self.enc_mu_logsig.weight)

        # Xavier intiialization for the weights of all the layers in the encoder
        for layer in self.encoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        # Xavier intialization for the weights of all the layers in the decoder
        for layer in self.decoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + std * eps
        

    def posterior(self, x:Tensor) -> Distribution:
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)
        mu = self.enc_mu_logsig(mu)
        log_sigma = self.enc_mu_logsig(log_sigma)
        return mu, log_sigma

    def forward(self, x):
        mu, log_sigma = self.posterior(x)
        z = self.reparameterize(mu, log_sigma)
        reconstruction = self.decoder(z)
        return reconstruction, z, mu, log_sigma

# Define the loss function

def loss_function(reconstruction, x, mu, log_sigma, beta):
    reconstruction_loss = functional.mse_loss(reconstruction, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    elbo = reconstruction_loss + beta*kl_loss
    return elbo, kl_loss*beta, reconstruction_loss


######################################
## --- Training and evaluation ---  ##
######################################

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


# Initialization of the model, evaluator and optimizer

# VAE
latent_features = 256
print(f'Shape of the archs4 dataset (hd5): {archs4_train[0].shape}')
print(f'Shape of the gtex dataset (hd5): {gtex_test[0][0].shape}')
vae = VariationalAutoencoder(archs4_train[0].shape, latent_features)

# Beta for the loss function
beta = 1

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

# Define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

# Initialize training loop
epoch = 0
num_epochs = 100

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f">> Using device: {device}")

# Move the model to the device
vae = vae.to(device)

# Initialize list to store losses
all_training_losses = []
all_validation_losses = []

# Define the number of samples to print and save
num_samples = 5  
num_proteins = 10

# Training
while epoch < num_epochs:

    epoch += 1
    training_epoch_data = defaultdict(list)
    vae.train()

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
        # pseudocount = 1e-8
        # x = x + pseudocount

        # perform a forward pass through the model 
        reconstruction, z, mu, log_sigma = vae(x)

        # Get the loss and that stuff
        elbo, kl_loss, reconstruction_loss = loss_function(reconstruction, x, mu, log_sigma, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the losses for each iteration
        all_training_losses.append(loss.item())

        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]

    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()

        # Just load a single batch from the test loader
        x, y = next(iter(gtex_test_dataloader))
        x = x.to(device)
        pseudocount = 1e-8
        x = x + pseudocount

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        # Accumulate the losses for each iteration
        all_validation_losses.append(loss.item())

        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
        
        # Get the reconstructions
        reconstructed = vae(x)['px'].sample().view(-1, *vae.input_shape).cpu().numpy()

        # Print and save original and reconstructed data to a text file
        with open('../Log_out_files/original_and_reconstructed_LN_baseline_APM.txt', 'a') as file:
            file.write(f"Epoch [{epoch}/{num_epochs}]\n")

            # Print and store a subset of samples
            for i in range(num_samples):
                original_sample = x[i][:num_proteins].squeeze().cpu().numpy()
                reconstructed_sample = reconstructed[i][:num_proteins].squeeze()

                # Print the original and reconstructed data
                print(f"Sample {i+1} - Original: {original_sample}")
                print(f"Sample {i+1} - Reconstructed: {reconstructed_sample}")

                # Write the original and reconstructed data to the text file
                file.write(f"Sample {i+1} - Original:\n")
                file.write(f"{original_sample}\n")
                file.write(f"Sample {i+1} - Reconstructed:\n")
                file.write(f"{reconstructed_sample}\n\n")

    # Generate plots every desired interval
    if epoch % 5 == 0:

        # Plot ELBO and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
        ax.plot(training_data['elbo'], label='Training')
        ax.plot(validation_data['elbo'], label='Validation')
        ax.legend()
        fig.savefig('../plots/elbo_plot_LN_baseline_APM.png')
        plt.close(fig)

        # Plot KL and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
        ax.plot(training_data['kl'], label='Training')
        ax.plot(validation_data['kl'], label='Validation')
        ax.legend()
        fig.savefig('../plots/kl_plot_LN_baseline_APM.png')
        plt.close(fig)

        # Plot NLL and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
        ax.plot(training_data['log_px'], label='Training')
        ax.plot(validation_data['log_px'], label='Validation')
        ax.legend()
        fig.savefig('../plots/nll_plot_LN_baseline_APM.png')
        plt.close(fig)

        # Plot the training loss values across iterations and save as PNG
        fig, ax = plt.subplots()
        ax.set_title('Training Loss across Iterations')
        ax.plot(all_training_losses, label='Training Loss')
        ax.legend()
        fig.savefig('../plots/loss_plot_LN_baseline_APM.png')
        plt.close(fig)


    # except Exception as e:
    #     print(f"Error in dataloader (last batch being less than {train_batch_size}): {e}")
    #     continue

    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    # make_vae_plots(vae, x, outputs, training_data, validation_data)

print('\nMetrics calculation:')

vae_path = "../VAE_settings/vae_settings_LN_baseline_APM.pth"
encoder_path = "../VAE_settings/encoder_LN_baseline_APM.pth"
decoder_path = "../VAE_settings/decoder_LN_baseline_APM.pth"

torch.save(vae.state_dict(), vae_path)
torch.save(vae.encoder.state_dict(), encoder_path)
torch.save(vae.decoder.state_dict(), decoder_path)

print('\nPlots representation:')

# Plot ELBO and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
ax.plot(training_data['elbo'], label='Training')
ax.plot(validation_data['elbo'], label='Validation')
ax.legend()
fig.savefig('../plots/elbo_plot_LN_baseline_APM.png')
plt.close(fig)

# Plot KL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
ax.plot(training_data['kl'][5:], label='Training')
ax.plot(validation_data['kl'][5:], label='Validation')
ax.legend()
fig.savefig('../plots/kl_plot_LN_baseline_APM.png')
plt.close(fig)

# Plot NLL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
ax.plot(training_data['log_px'], label='Training')
ax.plot(validation_data['log_px'], label='Validation')
ax.legend()
fig.savefig('../plots/nll_plot_LN_baseline_APM.png')
plt.close(fig)

# Plot the training loss values across iterations and save as PNG
fig, ax = plt.subplots()
ax.set_title('Training Loss across Iterations')
ax.plot(all_training_losses[100:], label='Training Loss')
ax.legend()
fig.savefig('../plots/loss_plot_LN_baseline_APM.png')
plt.close(fig)

