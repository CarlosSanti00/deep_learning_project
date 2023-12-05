import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader
# Import IsoDatasets if it's a custom module

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

# Load data from archs4_train dataset
archs4_train_data = []
for batch in archs4_train_dataloader:
    data = batch  # Access the data from the batch (modify if needed based on your dataset structure)
    archs4_train_data.extend(data.numpy())  # Assuming the data is in a NumPy array format

# Perform statistical analysis
mean = np.mean(archs4_train_data)
median = np.median(archs4_train_data)
std_dev = np.std(archs4_train_data)
skewness = stats.skew(archs4_train_data)
kurtosis = stats.kurtosis(archs4_train_data)

# Create histogram and fit distributions for comparison
plt.figure(figsize=(8, 6))
plt.hist(archs4_train_data, bins=50, density=True, alpha=0.6, color='g')

dist_names = ['norm', 'lognorm', 'expon', 'gamma']  # Example distributions
for dist_name in dist_names:
    dist = getattr(stats, dist_name)
    params = dist.fit(archs4_train_data)
    pdf = dist.pdf(np.linspace(np.min(archs4_train_data), np.max(archs4_train_data), 100), *params)
    plt.plot(np.linspace(np.min(archs4_train_data), np.max(archs4_train_data), 100), pdf, label=dist_name)

plt.legend()
plt.title('Histogram and Fitted Distributions')
plt.xlabel('Values')
plt.ylabel('Density')

# Save the plot
plt.savefig('histogram_and_distributions.png')
plt.show()

# Perform statistical tests (e.g., Kolmogorov-Smirnov test)
_, p_value = stats.kstest(archs4_train_data, 'norm')
if p_value > 0.05:  # Assuming significance level of 0.05
    print("Data follows a normal distribution.")
else:
    print("Data does not follow a normal distribution.")

# Perform statistical tests (e.g., Kolmogorov-Smirnov test)
_, p_value = stats.kstest(archs4_train_data, 'lognorm')
if p_value > 0.05:  # Assuming significance level of 0.05
    print("Data follows a lognormal distribution.")
else:
    print("Data does not follow a lognormal distribution.")

# Perform statistical tests (e.g., Kolmogorov-Smirnov test)
_, p_value = stats.kstest(archs4_train_data, 'expon')
if p_value > 0.05:  # Assuming significance level of 0.05
    print("Data follows a exponential distribution.")
else:
    print("Data does not follow a exponential distribution.")

# Perform statistical tests (e.g., Kolmogorov-Smirnov test)
_, p_value = stats.kstest(archs4_train_data, 'gamma')
if p_value > 0.05:  # Assuming significance level of 0.05
    print("Data follows a gamma distribution.")
else:
    print("Data does not follow a gamma distribution.")
