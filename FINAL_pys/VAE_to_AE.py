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

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        epsilon = self.sample_epsilon()
        z = self.mu + self.sigma * epsilon
        return z

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return -0.5 * ((z - self.mu) / self.sigma).pow(2) - 0.5 * math.log(2 * math.pi) - self.sigma.log()

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a LogNormal observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape: torch.Size, latent_features: int, use_baseline: bool = False) -> None:
        super(VariationalAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        print(self.observation_features)

        # Inference Network
        if use_baseline:
            # Use the baseline structure
            self.encoder = nn.Sequential(
                nn.Linear(in_features=self.observation_features, out_features=2 * latent_features),  # <- note the 2*latent_features
                nn.ReLU()  # Add ReLU activation after the last linear layer
            )
        else:
            # Use the original structure
            self.encoder = nn.Sequential(
                nn.Linear(in_features=self.observation_features, out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=2 * latent_features),  # <- note the 2*latent_features
                nn.ReLU()
            )

        # Generative Model
        if use_baseline:
            # Use the baseline structure
            self.decoder = nn.Sequential(
                nn.Linear(in_features=latent_features, out_features=self.observation_features)
            )
        else:
            # Use the original structure
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

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * latent_features])))

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)

        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_params = self.decoder(z)
        px_params = px_params.view(-1, *self.input_shape)
        return LogNormal(px_params, 1.0)  # Assuming unit variance


    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        x = x.view(x.size(0), -1)

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}


    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)

        # sample the prior
        z = pz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'z': z}

# Module for variational inference
def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=0.):
        super().__init__()
        self.beta = beta

    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px
        beta_elbo = log_px

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

## --- Training and evaluation ---

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
baseline = VariationalAutoencoder(archs4_train[0].shape, latent_features, use_baseline=True)

# Evaluator: Variational Inference
beta = 0
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
baseline_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

# Define dictionary to store the training curves
training_data = defaultdict(list)
baseline_data = defaultdict(list)
validation_data = defaultdict(list)
validation_baseline_data = defaultdict(list)

# Initialize training loop
epoch = 0
num_epochs = 100

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f">> Using device: {device}")

# Move the model to the device
vae = vae.to(device)
baseline = baseline.to(device)

# Initialize list to store losses
all_training_losses = []
all_training_baseline_losses = []
all_validation_losses = []
all_validation_baseline_losses = []

# Define the number of samples to print and save
num_samples = 5  
num_proteins = 10

# Training
while epoch < num_epochs:

    epoch += 1
    training_epoch_data = defaultdict(list)
    baseline_epoch_data = defaultdict(list)
    vae.train()
    baseline.train()

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

        # perform a forward pass through the model and compute the ELBO
        vae_loss, vae_diagnostics, vae_outputs = vi(vae, x)
        baseline_loss, baseline_diagnostics, baseline_outputs = vi(baseline, x)

        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

        baseline_optimizer.zero_grad()
        baseline_loss.backward()
        baseline_optimizer.step()

        # Accumulate the losses for each iteration
        all_training_losses.append(vae_loss.item())
        all_training_baseline_losses.append(baseline_loss.item())

        # gather data for the current bach
        for k, v in vae_diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]

        # gather data for the Baseline
        for k, v in baseline_diagnostics.items():
            baseline_epoch_data[k] += [v.mean().item()]

    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # gather data for the full epoch (baseline)
    for k, v in baseline_epoch_data.items():
        baseline_data[k] += [np.mean(baseline_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()
        baseline.eval()

        # Just load a single batch from the test loader
        x, y = next(iter(gtex_test_dataloader))
        x = x.to(device)
        pseudocount = 1e-8
        x = x + pseudocount

        # perform a forward pass through the models and compute the ELBO
        vae_loss, vae_diagnostics, vae_outputs = vi(vae, x)
        baseline_loss, baseline_diagnostics, baseline_outputs = vi(baseline, x)

        # Accumulate the losses for each iteration
        all_validation_losses.append(vae_loss.item())
        all_validation_baseline_losses.append(baseline_loss.item())

        # gather data for the validation step for VAE
        for k, v in vae_diagnostics.items():
            validation_data[k] += [v.mean().item()]

        # gather data for the validation step for Baseline
        for k, v in baseline_diagnostics.items():
            validation_baseline_data[k] += [v.mean().item()]

        # Get the reconstructions for both VAE and baseline
        reconstructed_vae = vae(x)['px'].sample().view(-1, *vae.input_shape).cpu().numpy()
        reconstructed_baseline = baseline(x)['px'].sample().view(-1, *baseline.input_shape).cpu().numpy()

        # Print and save original and reconstructed data to a text file
        # with open('../Log_out_files/original_and_reconstructed_LN(final)_CSL.txt', 'a') as file:
        #     file.write(f"Epoch [{epoch}/{num_epochs}]\n")

        #     # Print and store a subset of samples for VAE
        #     for i in range(num_samples):
        #         original_sample = x[i][:num_proteins].squeeze().cpu().numpy()
        #         reconstructed_sample = reconstructed_vae[i][:num_proteins].squeeze()

        #         # Print the original and reconstructed data
        #         print(f"Sample {i+1} - Original (VAE): {original_sample}")
        #         print(f"Sample {i+1} - Reconstructed (VAE): {reconstructed_sample}")

        #         # Write the original and reconstructed data to the text file
        #         file.write(f"Sample {i+1} - Original (VAE):\n")
        #         file.write(f"{original_sample}\n")
        #         file.write(f"Sample {i+1} - Reconstructed (VAE):\n")
        #         file.write(f"{reconstructed_sample}\n\n")

        #     # Print and store a subset of samples for Baseline
        #     for i in range(num_samples):
        #         original_sample = x[i][:num_proteins].squeeze().cpu().numpy()
        #         reconstructed_sample = reconstructed_baseline[i][:num_proteins].squeeze()

        #         # Print the original and reconstructed data
        #         print(f"Sample {i+1} - Original (Baseline): {original_sample}")
        #         print(f"Sample {i+1} - Reconstructed (Baseline): {reconstructed_sample}")

        #         # Write the original and reconstructed data to the text file
        #         file.write(f"Sample {i+1} - Original (Baseline):\n")
        #         file.write(f"{original_sample}\n")
        #         file.write(f"Sample {i+1} - Reconstructed (Baseline):\n")
        #         file.write(f"{reconstructed_sample}\n\n")

    # Generate plots every desired interval
    if epoch % 5 == 0:

        # Plot ELBO and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
        ax.plot(training_data['elbo'], label='VAE Training')
        ax.plot(validation_data['elbo'], label='VAE Validation')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_elbo.png')
        plt.close(fig)

        # Plot ELBO and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
        ax.plot(baseline_data['elbo'], label='Baseline Training')
        ax.plot(validation_baseline_data['elbo'], label='Baseline Validation')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_elbo_baseline_CSL.png')
        plt.close(fig)

        # Plot KL and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
        ax.plot(training_data['kl'], label='VAE Training')
        ax.plot(validation_data['kl'], label='VAE Validation')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_kl.png')
        plt.close(fig)

        # Plot KL and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
        ax.plot(baseline_data['kl'], label='Baseline Training')
        ax.plot(validation_baseline_data['kl'], label='Baseline Validation')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_kl_baseline_CSL.png')
        plt.close(fig)

        # Plot NLL and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
        ax.plot(training_data['log_px'], label='VAE Training')
        ax.plot(validation_data['log_px'], label='VAE Validation')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_elbo_NLL.png')
        plt.close(fig)

        # Plot NLL and save as PNG
        fig, ax = plt.subplots()
        ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
        ax.plot(baseline_data['log_px'], label='Baseline Training')
        ax.plot(validation_baseline_data['log_px'], label='Baseline Validation')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_NLL_baseline.png')
        plt.close(fig)

        # Plot the training loss values across iterations and save as PNG
        fig, ax = plt.subplots()
        ax.set_title('Training Loss across Iterations')
        ax.plot(all_training_losses, label='VAE Training Loss')
        ax.plot(all_validation_losses, label='VAE Validation Loss')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_loss_plot.png')
        plt.close(fig)

        # Plot the baseline loss values across iterations and save as PNG
        fig, ax = plt.subplots()
        ax.set_title('Validation Loss across Iterations')
        ax.plot(all_training_baseline_losses, label='Baseline Training Loss')
        ax.plot(all_validation_baseline_losses, label='Baseline Validation Loss')
        ax.legend()
        fig.savefig('../plots/VAE_to_AE_loss_plot_baseline.png')
        plt.close(fig)


    # except Exception as e:
    #     print(f"Error in dataloader (last batch being less than {train_batch_size}): {e}")
    #     continue

    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    # make_vae_plots(vae, x, outputs, training_data, validation_data)


vae_path = "../VAE_settings/VAE_to_AE.pth"
# encoder_path = "../VAE_settings/encoder_LN(final)_CSL.pth"
# decoder_path = "../VAE_settings/decoder_LN(final)_CSL.pth"

# Save the entire model (including parameters and architecture)
torch.save(vae, vae_path)

# torch.save(vae.encoder.state_dict(), encoder_path)
# torch.save(vae.decoder.state_dict(), decoder_path)

print('\nPlots representation:')

# Plot ELBO and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
ax.plot(training_data['elbo'], label='VAE Training')
ax.plot(validation_data['elbo'], label='VAE Validation')
ax.legend()
fig.savefig('../plots/VAE_to_AE_elbo.png')
plt.close(fig)

# Plot ELBO and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
ax.plot(baseline_data['elbo'], label='Baseline Training')
ax.plot(validation_baseline_data['elbo'], label='Baseline Validation')
ax.legend()
fig.savefig('../plots/VAE_to_AE_elbo_baseline.png')
plt.close(fig)

# Plot KL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
ax.plot(training_data['kl'][5:], label='VAE Training')
ax.plot(validation_data['kl'][5:], label='VAE Validation')
ax.legend()
ax.set_xticks(range(5, num_epochs + 1, 10))
fig.savefig('../plots/VAE_to_AE_KL.png')
plt.close(fig)

# Plot KL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
ax.plot(baseline_data['kl'], label='Baseline Training')
ax.plot(validation_baseline_data['kl'], label='Baseline Validation')
ax.legend()
ax.set_xticks(range(5, num_epochs + 1, 10))
fig.savefig('../plots/VAE_to_AE_KL_baseline.png')
plt.close(fig)

# Plot NLL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
ax.plot(training_data['log_px'], label='VAE Training')
ax.plot(validation_data['log_px'], label='VAE Validation')
ax.legend()
fig.savefig('../plots/VAE_to_AE_NLL.png')
plt.close(fig)

# Plot NLL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
ax.plot(baseline_data['log_px'], label='Baseline Training')
ax.plot(validation_baseline_data['log_px'], label='Baseline Validation')
ax.legend()
fig.savefig('../plots/VAE_to_AE_NLL_baseline.png')
plt.close(fig)

# Plot the training loss values across iterations and save as PNG
fig, ax = plt.subplots()
ax.set_title('Training and Validation Loss across Iterations')
ax.plot(all_training_losses[100:], label='Training Loss')
ax.plot(all_validation_losses[100:], label='Validation Loss')
ax.legend()
ax.set_xticks(range(100, len(all_training_losses)+1, 100))
fig.savefig('../plots/VAE_to_AE_loss_plot.png')
plt.close(fig)

# Plot the training loss values across iterations and save as PNG
fig, ax = plt.subplots()
ax.set_title('Training and Validation Loss of the Baseline across Iterations')
ax.plot(all_training_baseline_losses[100:], label='Baseline Training Loss')
ax.plot(all_validation_baseline_losses[100:], label='Baseline Validation Loss')
ax.legend()
ax.set_xticks(range(100, len(all_training_baseline_losses) + 1, 100))
fig.savefig('../plots/VAE_to_AE_loss_plot_baseline.png')
plt.close(fig)

# Create an h5py file and pass the gtex_data through the encoder of the VAE
output_file_path = "../VAE_settings/latent_features_VAE_to_AE.h5"

# Load the gtex_data using the DataLoader
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/")
gtex_test_dataloader = DataLoader(gtex_test, batch_size=eval_batch_size, shuffle=False)


with h5py.File(output_file_path, 'w') as hf:

    latent_features_dataset = hf.create_dataset('latent_features', shape=(len(gtex_test), latent_features), dtype='float32')

    # Encode and save latent features in batches
    start_index = 0

    with torch.no_grad():
        
        vae.to(device)
        vae.eval()

        for x, y in gtex_test_dataloader:
            x = x.to(device)
            pseudocount = 1e-8
            x = x + pseudocount

            # Get the latent features from the encoder
            qz = vae.posterior(x)
            latent_features = qz.mu.cpu().numpy()  # Assuming you want to use the mean of the latent features

            # Save the latent features in the h5py file
            latent_features_dataset[start_index:start_index + len(latent_features)] = latent_features

            start_index += len(latent_features)

print(f"Latent features saved to {output_file_path}")

# # Save parameters in txt files
# print('\nSaving metrics file:')
# np.savetxt('metrics_VAE/Training_ELBO.txt', training_data['elbo'])
# np.savetxt('metrics_VAE/Training_baseline_ELBO.txt', baseline_data['elbo'])
# np.savetxt('metrics_VAE/Validation_ELBO.txt', validation_data['elbo'])
# np.savetxt('metrics_VAE/Validation_baseline_ELBO.txt', validation_baseline_data['elbo'])

# np.savetxt('metrics_VAE/Training_KL.txt', training_data['kl'])
# np.savetxt('metrics_VAE/Training_baseline_KL.txt', baseline_data['kl'])
# np.savetxt('metrics_VAE/Validation_KL.txt', validation_data['kl'])
# np.savetxt('metrics_VAE/Validation_baseline_KL.txt', validation_baseline_data['kl'])

# np.savetxt('metrics_VAE/Training_nLL.txt', training_data['log_px'])
# np.savetxt('metrics_VAE/Training_baseline_nLL.txt', baseline_data['log_px'])
# np.savetxt('metrics_VAE/Validation_nLL.txt', validation_data['log_px'])
# np.savetxt('metrics_VAE/Validation_baseline_nLL.txt', validation_baseline_data['log_px'])

# np.savetxt('metrics_VAE/Training_loss.txt', all_training_losses)
# np.savetxt('metrics_VAE/Training_baseline_loss.txt', all_training_baseline_losses)
# np.savetxt('metrics_VAE/Validation_loss.txt', all_training_baseline_losses)
# np.savetxt('metrics_VAE/Validation_baseline_loss.txt', all_validation_baseline_losses)