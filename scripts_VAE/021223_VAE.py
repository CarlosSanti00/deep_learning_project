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
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.distributions import Normal
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
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        print(self.observation_features)


        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=2048),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=512, out_features=2*latent_features), # <- note the 2*latent_features
            #nn.Sigmoid()  # Sigmoid activation to the last layer
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=2048, out_features=self.observation_features)
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

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
        px_mu = self.decoder(z)
        px_mu = px_mu.view(-1, *self.input_shape) # reshape the output
        px_log_sigma = torch.ones_like(px_mu)
        return Normal(loc=px_mu, scale=torch.exp(px_log_sigma))


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
    def __init__(self, beta:float=1.):
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
        elbo = log_px - kl
        beta_elbo = log_px - (self.beta*kl)

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

## --- Training and evaluation ---

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

# Evaluator: Variational Inference
beta = 1
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

# Initialize training loop
epoch = 0
num_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x in archs4_train_dataloader:

        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

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

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        # Accumulate the losses for each iteration
        all_validation_losses.append(loss.item())

        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]

         # Get the reconstructions
        reconstructed = vae(x)['px'].probs.view(-1, *vae.input_shape).cpu().numpy()

        # Print and save original and reconstructed data to a text file
        with open('../Log_out_files/original_and_reconstructed.txt', 'a') as file:
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


    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    # make_vae_plots(vae, x, outputs, training_data, validation_data)

vae_path = "../VAE_settings/vae_settings02.pth"
encoder_path = "../VAE_settings/encoder02.pth"
decoder_path = "../VAE_settings/encoder02.pth"

torch.save(vae.state_dict(), vae_path)
torch.save(vae.encoder.state_dict(), encoder_path)
torch.save(vae.decoder.state_dict(), decoder_path)


# Plot ELBO and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
ax.plot(training_data['elbo'], label='Training')
ax.plot(validation_data['elbo'], label='Validation')
ax.legend()
fig.savefig('../plots/elbo_plot02.png')
plt.close(fig)

# Plot KL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
ax.plot(training_data['kl'], label='Training')
ax.plot(validation_data['kl'], label='Validation')
ax.legend()
fig.savefig('../plots/kl_plot02.png')
plt.close(fig)

# Plot NLL and save as PNG
fig, ax = plt.subplots()
ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
ax.plot(training_data['log_px'], label='Training')
ax.plot(validation_data['log_px'], label='Validation')
ax.legend()
fig.savefig('../plots/nll_plot02.png')
plt.close(fig)

# Plot the training loss values across iterations and save as PNG
fig, ax = plt.subplots()
ax.set_title('Training Loss across Iterations')
ax.plot(all_training_losses, label='Training Loss')
ax.plot(all_validation_losses, label='TValidation Loss')
ax.legend()
fig.savefig('../plots/loss_plot02.png')
plt.close(fig)

