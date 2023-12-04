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

try:
    from plotting import plot_autoencoder_stats
except Exception as ex:
    print(f"If using Colab, you may need to upload `plotting.py`. \
          \nIn the left pannel, click `Files > upload to session storage` and select the file `plotting.py` from your computer \
          \n---------------------------------------------")
    print(ex)

cuda = torch.cuda.is_available()

# define size variables
num_features = 18965

class AutoEncoder(nn.Module):
    def __init__(self, hidden_units, latent_features=2):
        super(AutoEncoder, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=hidden_units),
            nn.ReLU(),
            # bottleneck layer
            nn.Linear(in_features=hidden_units, out_features=latent_features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=hidden_units),
            nn.ReLU(),
            # output layer, projecting back to image size
            nn.Linear(in_features=hidden_units, out_features=num_features)
        )

    def forward(self, x): 
        outputs = {}
        # we don't apply an activation to the bottleneck layer
        z = self.encoder(x)
        
        # apply sigmoid to output to get pixel intensities between 0 and 1
        x_hat = torch.sigmoid(self.decoder(z))
        
        return {
            'z': z,
            'x_hat': x_hat
        }


# Choose the shape of the autoencoder
net = AutoEncoder(hidden_units=128, latent_features=2)

if cuda:
    net = net.cuda()

print(net)
# if you want L2 regularization, then add weight_decay to SGD
optimizer = optim.SGD(net.parameters(), lr=0.25)

# We will use pixel wise mean-squared error as our loss function
loss_function = nn.MSELoss()


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



# Define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

 

# Initialize training loop

num_epochs = 100

train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    batch_loss = []
    net.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x in archs4_train_dataloader:
        
        if cuda:
            x = x.cuda()
        
        outputs = net(x)
        x_hat = outputs['x_hat']

        # note, target is the original tensor, as we're working with auto-encoders
        loss = loss_function(x_hat, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())

    train_loss.append(np.mean(batch_loss))

    # Evaluate, do not propagate gradients
    with torch.no_grad():
        net.eval()
        
        # Just load a single batch from the test loader
        x, y = next(iter(gtex_test_dataloader))
        
        if cuda:
            x = x.cuda()
        
        outputs = net(x)

        # We save the latent variable and reconstruction for later use
        # we will need them on the CPU to plot
        x_hat = outputs['x_hat']
        z = outputs['z'].cpu().numpy()

        loss = loss_function(x_hat, x)

        valid_loss.append(loss.item())
    
    if epoch == 0:
        continue

    # live plotting of the trainig curves and representation
    print("x_hat:", x_hat.cpu() , "z:", z , "y:", y, "train_loss:", train_loss, "valid_loss:", valid_loss , "epoch:", epoch)
   # plot_autoencoder_stats(x=x.cpu(),
    #                       x_hat=x_hat.cpu(),
    #                       z=z,
     #                      y=y,
      #                     train_loss=train_loss,
       #                    valid_loss=valid_loss,
        #                   epoch=epoch,
         #                  dimensionality_reduction_op=None)
    

	 # -- Plotting --
    f, axarr = plt.subplots(2, 2, figsize=(20, 20))

    # Loss
    ax = axarr[0, 0]
    ax.set_title("Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')

    ax.plot(np.arange(epoch + 1), train_loss, color="black")
    ax.plot(np.arange(epoch + 1), valid_loss, color="gray", linestyle="--")
    ax.legend(['Training error', 'Validation error'])

    # Latent space
    ax = axarr[0, 1]

    ax.set_title('Latent space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.savefig('model.png')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")
