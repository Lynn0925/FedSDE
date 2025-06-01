import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DatasetFolder
import numpy as np
from collections import defaultdict
import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logger()

def partition_data(train_dataset, test_dataset, num_clients, private_data_ratio, non_iid_alpha, batch_size):
    """
    Partition the dataset into client private data, client local public data, and a global public dataset.
    Applies Dirichlet distribution for Non-IID partitioning of private data.
    """
    num_train_samples = len(train_dataset)
    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))

    # Determine split sizes
    num_private_total = int(num_train_samples * private_data_ratio)
    num_public_total = num_train_samples - num_private_total

    # Split for global public dataset
    indices = np.arange(num_train_samples)
    np.random.shuffle(indices)

    global_public_indices = indices[:num_public_total]
    client_train_indices = indices[num_public_total:]

    global_public_dataset = Subset(train_dataset, global_public_indices)
    # Global public dataloader for server-side KD
    global_public_dataloader = DataLoader(global_public_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Global Public Dataset Size: {len(global_public_dataset)}")


    # Partition client training data (Non-IID using Dirichlet)
    min_size = 0
    min_req_samples_per_client = 100 # Minimum samples per client for training
    client_indices = [[] for _ in range(num_clients)]
    while min_size < min_req_samples_per_client:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels[client_train_indices] == k)[0] # Indices of class k in client_train_indices
            np.random.shuffle(idx_k)
            # Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(non_iid_alpha, num_clients))
            proportions = [p * (len(idx_k) / sum(proportions)) for p in proportions]
            for i in range(num_clients):
                idx_batch[i] += idx_k[int(sum(proportions[:i])):int(sum(proportions[:i+1]))].tolist()

        min_size = min(len(idx_j) for idx_j in idx_batch)
        if min_size < min_req_samples_per_client: # Resample if any client gets too few samples
            logger.warning(f"Resampling data partitions due to small client size: {min_size}")

    for i in range(num_clients):
        np.random.shuffle(idx_batch[i])
        client_indices[i] = client_train_indices[idx_batch[i]].tolist()

    client_private_datasets = []
    client_public_local_datasets = [] # For optional local public data for clients

    for i in range(num_clients):
        # Here we assume client_private_datasets are the primary training data
        # For FedSDE, we might not have 'local public data' per client,
        # but rather just the global public data for generating soft labels.
        # This split can be adjusted based on exact requirements.
        client_private_datasets.append(DataLoader(Subset(train_dataset, client_indices[i]), batch_size=batch_size, shuffle=True))
        # client_public_local_datasets will be empty in this simple setup or can be a subset of global_public_dataset
        # For simplicity, we'll just pass the global_public_dataloader to clients for soft label generation
        client_public_local_datasets.append(None) # Not used directly for local public data split

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Total Private Data Samples for Clients: {sum(len(d.dataset) for d in client_private_datasets)}")
    logger.info(f"Number of clients: {num_clients}")

    return client_private_datasets, client_public_local_datasets, global_public_dataloader, test_dataloader