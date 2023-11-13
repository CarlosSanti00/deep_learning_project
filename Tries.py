from hdf5_scripts import IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Example of making a training set that excludes samples from the brain and a test set with only samples from the brain
# If you have enough memory, you can load the dataset to memory using the argument load_in_mem=True
gtex_train = IsoDatasets.GtexDataset("/zhome/5a/4/181325/deep_learning/hdf5_scripts/hdf5_data/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/zhome/5a/4/181325/deep_learning/hdf5_scripts/hdf5_data/", include='brain')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))