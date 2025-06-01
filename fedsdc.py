#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
from collections import defaultdict
from json import JSONEncoder

import torch
import numpy as np # Ensure numpy is imported for seeding
from tqdm import tqdm

# Import utilities for logging, arguments, config, and data partitioning
from utils import Logger, fed_args, read_config, log_config,partition_data # Added partition_data

# Import existing baseline clients and servers (as provided by the user)
from fed_baselines.client_base import EnsembleClient
from fed_baselines.client_awesome import AwesomeClient
from fed_baselines.server_base import EnsembleServer
from fed_baselines.client_cvae import FedCVAEClient
from fed_baselines.server_cvae import FedCVAEServer
from fed_baselines.server_dense import DENSEServer
from fed_baselines.server_coboost import CoBoostServer
from fed_baselines.server_dafl import DAFLServer
from fed_baselines.server_adi import ADIServer
from fed_baselines.server_central import CentralServer
from fed_baselines.server_fedavg import FedAVGServer
from fed_baselines.server_awesome import FedAWSServer

# >>>>>>>>>>>>>>>>>> FedSDE Specific Code: New Import Paths <<<<<<<<<<<<<<<<<<<<
# Import FedSDE client and server classes from their new locations in fed_baselines
from fed_baselines.client_sde import FedSDEClient
from fed_baselines.server_sde import FedSDEServer
#from src.models import ClassifierModel, DiffusionModel # Models still live in src/models
# <<<<<<<<<<<<<<<<<< FedSDE Specific Code: New Import Paths >>>>>>>>>>>>>>>>>>>>

from postprocessing.recorder import Recorder
# Original baseline dataloaders (might be partially replaced or complemented)
from preprocessing.baselines_dataloader import divide_data, divide_data_with_dirichlet, divide_data_with_local_cls, load_data
from utils.models import * # Assumed to import common classifier models like LeNet, ResNet, etc.

json_types = (list, dict, str, int, float, bool, type(None))

# Initialize arguments
args = fed_args()
args = read_config(args.config, args)

# >>>>>>>>>>>>>>>>>> FedSDE Specific Code: Add FedSDE Arguments <<<<<<<<<<<<<<<<<<<<
# Add FedSDE specific arguments, if they are not already in your config.yaml
# This ensures default values are set if not provided externally.
parser_fedsde = argparse.ArgumentParser(add_help=False) # Use add_help=False to merge later
parser_fedsde.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
parser_fedsde.add_argument('--kl_beta', type=float, default=0.5, help='Beta for CE loss in server-side KD')
parser_fedsde.add_argument('--num_hard_samples', type=int, default=100, help='Number of hard samples to generate per client')
parser_fedsde.add_argument('--diff_epochs', type=int, default=10, help='Number of local epochs for diffusion model training')
parser_fedsde.add_argument('--diff_lr', type=float, default=0.001, help='Learning rate for diffusion model')
parser_fedsde.add_argument('--private_data_ratio', type=float, default=0.8, help='Ratio of private data for clients (remaining is public local)')
parser_fedsde.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs for classifier training')
parser_fedsde.add_argument('--batch_size', type=int, default=64, help='Batch size for local training')
parser_fedsde.add_argument('--lr', type=float, default=0.01, help='Learning rate for classifier')

# Merge FedSDE-specific arguments into the main args object
temp_fedsde_args, _ = parser_fedsde.parse_known_args()
for arg_name, arg_value in vars(temp_fedsde_args).items():
    if not hasattr(args, arg_name): # Only add if it doesn't already exist from config
        setattr(args, arg_name, arg_value)
# <<<<<<<<<<<<<<<<<< FedSDE Specific Code: Add FedSDE Arguments >>>>>>>>>>>>>>>>>>>>


# Setup logger
if Logger.logger is None:
    logger = Logger()
    os.makedirs("train_records/", exist_ok=True)
    logger.set_log_name(os.path.join("train_records", f"train_record_{args.save_name}.log"))
    logger = logger.get_logger()
    log_config(args)

using_wandb = args.using_wandb

# Clear GPU cache
torch.cuda.empty_cache()

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def fed_run():
    """
    Main function for the baselines of federated learning
    """

    # Add FedSDE to the list of supported algorithms
    # >>>>>>>>>>>>>>>>>> FedSDE Specific Code: Update algo_list <<<<<<<<<<<<<<<<<<<<
    algo_list = ["FedCVAE", "DENSE", "ENSEMBLE", "CoBoost", "DAFL", "ADI", "Central", "FedAVG", "Awesome", "FedSDE"]
    # <<<<<<<<<<<<<<<<<< FedSDE Specific Code: Update algo_list >>>>>>>>>>>>>>>>>>>>
    assert args.client_instance in algo_list, f"The federated learning algorithm '{args.client_instance}' is not supported"

    dataset_list = ["CIFAR10", 'TINYIMAGENET', "CIFAR100", 'Imagenette', "openImg", "COVID","SVHN", "MNIST"] # Added MNIST for example
    assert args.sys_dataset in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN", "Conv4",
                  "Conv5", "Conv6"]
    assert args.sys_model in model_list, "The model is not supported"

    random.seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.sys_i_seed)
    torch.set_num_threads(3)

    client_dict = {}
    recorder = Recorder()

    logger.info('======================Setup Clients and Data==========================')

    # >>>>>>>>>>>>>>>>>> FedSDE Specific Code: Data Loading & Pre-processing <<<<<<<<<<<<<<<<<<<<
    # Load full datasets first to ensure `partition_data` can work with them
    from torchvision import datasets, transforms
    if args.sys_dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_full_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        img_channels = 1
        img_size = 28
        num_classes = 10
    elif args.sys_dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_full_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_full_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        img_channels = 3
        img_size = 32
        num_classes = 10
    elif args.sys_dataset == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_full_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_full_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        img_channels = 3
        img_size = 32
        num_classes = 100
    elif args.sys_dataset == 'TINYIMAGENET':
        # Placeholder for TinyImageNet transforms and dataset loading
        # You'll need to implement this based on how TinyImageNet is structured
        raise NotImplementedError("TinyImageNet loading needs specific implementation.")
    else:
        raise NotImplementedError(f"Dataset {args.sys_dataset} is not configured for FedSDE data loading.")


    # Use the custom `partition_data` from `utils.py` for FedSDE specific data splits.
    # This function returns DataLoaders directly for clients and global public data.
    client_private_dataloaders, client_public_local_dataloaders, global_public_dataloader, test_dataloader_for_evaluation = \
        partition_data(train_full_dataset, test_full_dataset, args.sys_n_client,
                       args.private_data_ratio, args.sys_dataset_dir_alpha, args.batch_size)

    # For other baselines, use the original data loading/partitioning logic if needed.
    # This might create redundant data objects but ensures compatibility.
    trainset_config, testset_for_baselines, cls_record = divide_data_with_dirichlet(
        n_clients=args.sys_n_client,
        beta=args.sys_dataset_dir_alpha,
        dataset_name=args.sys_dataset,
        seed=args.sys_i_seed
    )
    # <<<<<<<<<<<<<<<<<< FedSDE Specific Code: Data Loading & Pre-processing >>>>>>>>>>>>>>>>>>>>


    logger.info('Clients in Total: %d' % len(trainset_config['users']))

    # Initialize the server based on the chosen FL algorithm
    logger.info('--------------------- Server Initialization ---------------------')
    if args.client_instance == 'FedCVAE':
        server = FedCVAEServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'DENSE':
        server = DENSEServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'ENSEMBLE':
        server = EnsembleServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'CoBoost':
        server = CoBoostServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'DAFL':
        server = DAFLServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'ADI':
        server = ADIServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'FedAVG':
        server = FedAVGServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'Awesome':
        server = FedAWSServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'FedSDE':
        # FedSDE Server needs a global classifier model and the global public dataloader
        server = FedSDEServer(
            args=args,
            client_ids=trainset_config['users'],  # Pass client IDs for base class constructor
            dataset_id=args.sys_dataset,
            model_name=args.sys_model,
            global_public_data_loader=global_public_dataloader# Pass the DataLoader for public data
        )
    # <<<<<<<<<<<<<<<<<< FedSDE Specific Code: Server Initialization >>>>>>>>>>>>>>>>>>>>
    else:
        raise NotImplementedError('Server initialization error for unknown algorithm.')

    # Load testset for server evaluation (common for all algos)
    # Use the test_dataloader generated by `partition_data` for consistency
    server.load_testset(test_dataloader_for_evaluation)
    server.load_cls_record(cls_record) # Assuming this is still relevant for some baselines/metrics

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    losses = defaultdict(list)
    logger.info('--------------------- Client Configuration and Local Training Stage ---------------------')
    
    # List to store soft labels uploaded by clients for FedSDE (will be aggregated by server)
    # >>>>>>>>>>>>>>>>>> FedSDE Specific Code <<<<<<<<<<<<<<<<<<<<
    uploaded_soft_labels_dict = {}
    generated_data_dict={}
    # <<<<<<<<<<<<<<<<<< FedSDE Specific Code >>>>>>>>>>>>>>>>>>>>

    # Loop through clients for initialization and local training
    for client_idx, client_id in enumerate(trainset_config['users']):
        if args.client_instance == 'FedCVAE':
            client_dict[client_id] = FedCVAEClient(args, client_id, epoch=args.client_instance_n_epoch,
                                                   dataset_id=args.sys_dataset, model_name=args.sys_model)
            server.client_dict[client_id] = client_dict[client_id]
            client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
            _, recon_loss, kld_loss = client_dict[client_id].train(client_dict[client_id].model)
            losses["recon_loss"].append(recon_loss)
            losses["kld_loss"].append(kld_loss)
        elif args.client_instance == 'Awesome':
            # Note: Original code had client_class = FedAWSServer, which is likely a typo.
            # Assuming it should be AwesomeClient.
            client_dict[client_id] = AwesomeClient(args, client_id, epoch=args.client_instance_n_epoch,
                                                  dataset_id=args.sys_dataset, model_name=args.sys_model)
            server.client_dict[client_id] = client_dict[client_id]
            client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
            server.client_model[client_id] = client_dict[client_id].model # Store client's model on server
            if args.client_model_root is not None and os.path.exists(
                    os.path.join(args.client_model_root, f"c{client_id}.pt")):
                weight = torch.load(os.path.join(args.client_model_root, f"c{client_id}.pt"), map_location="cpu")
                logger.info("Load Client {} from {}".format(client_id, os.path.join(args.client_model_root, f"c{client_id}.pt")))
                client_dict[client_id].model.load_state_dict(weight)
            else:
                c_loss = client_dict[client_id].train(client_dict[client_id].model)
                if not os.path.exists(os.path.join(args.sys_res_root, args.save_name)):
                    os.makedirs(os.path.join(args.sys_res_root, args.save_name))
                if args.save_client_model:
                    torch.save(client_dict[client_id].model.state_dict(), os.path.join(args.sys_res_root, args.save_name, f"c{client_id}.pt"))
                losses["client_loss"].append(c_loss)
        elif args.client_instance in ['DENSE', 'ENSEMBLE', "CoBoost", "DAFL", "ADI", "FedAVG"]:
            logger.info(f'--------------------- Client {client_id} Training stage ---------------------')
            client_class = EnsembleClient # This is the base client for these algos
            client_dict[client_id] = client_class(args, client_id, epoch=args.client_instance_n_epoch,
                                                  dataset_id=args.sys_dataset, model_name=args.sys_model)
            server.client_dict[client_id] = client_dict[client_id]
            client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
            server.client_model[client_id] = client_dict[client_id].model # Store client's model on server
            if args.client_model_root is not None and os.path.exists(
                    os.path.join(args.client_model_root, f"c{client_id}.pt")):
                logger.info('---------------------  Loading stage ---------------------')
                weight = torch.load(os.path.join(args.client_model_root, f"c{client_id}.pt"), map_location="cpu")
                logger.info("Load Client {} from {}".format(client_id, os.path.join(args.client_model_root, f"c{client_id}.pt")))
                client_dict[client_id].model.load_state_dict(weight)
            else:
                logger.info('---------------------  Training stage ---------------------')
                c_loss = client_dict[client_id].train(client_dict[client_id].model)
                if not os.path.exists(os.path.join(args.sys_res_root, args.save_name)):
                    os.makedirs(os.path.join(args.sys_res_root, args.save_name))
                if args.save_client_model:
                    torch.save(client_dict[client_id].model.state_dict(), os.path.join(args.sys_res_root, args.save_name, f"c{client_id}.pt"))
                losses["client_loss"].append(c_loss)
        # >>>>>>>>>>>>>>>>>> FedSDE Specific Code: Client Initialization & Local Task Execution <<<<<<<<<<<<<<<<<<
        elif args.client_instance == 'FedSDE':
            logger.info(f'--------------------- Client {client_id} FedSDE Local Stage ---------------------')
            # The `client_private_dataloaders` is a list of DataLoaders, indexed by client_idx
            private_data_loader_for_client = client_private_dataloaders[client_idx]

            client_dict[client_id] = FedSDEClient(
                args=args,
                client_id=client_id,
                epoch=args.local_epochs, # Use local_epochs for client training
                dataset_id=args.sys_dataset,
                model_name=args.sys_model,
                private_data_loader=private_data_loader_for_client, # Dataloader for client's private training data
                global_public_data_loader=global_public_dataloader, # Dataloader for public data (same for all clients)
                img_channels=img_channels,
                img_size=img_size,
                num_classes=num_classes
            )
            # Store client reference in server's client_dict for framework consistency if needed,
            # though FedSDE server primarily interacts via soft labels.
            server.client_dict[client_id] = client_dict[client_id]
            server.client_model[client_id] = client_dict[client_id].model

            logger.info(f"Client {client_id}: Executing FedSDE local tasks (diffusion, self-distillation, soft label generation)...")
            # Call the `train` method of FedSDEClient, which encapsulates the entire pipeline
            soft_labels ,hard_sample_loader= client_dict[client_id].train() # The train method of FedSDEClient returns soft_labels

            # Store soft labels for server aggregation later
            uploaded_soft_labels_dict[client_id] = soft_labels
            #generated_data_dict[client_id] = generated_hard_samples, hard_sample_soft_labels, hard_sample_pseudo_labels
            
        # <<<<<<<<<<<<<<<<<< FedSDE Specific Code: Client Initialization & Local Task Execution >>>>>>>>>>>>>>>>>>>>
        else:
            raise NotImplementedError("Client instance is not supported")

    for l_name, loss_list in losses.items():
        logger.info('--------------------- Client Training Ending ---------------------')
        logger.info('{}: {}'.format(l_name, [float(l) for l in loss_list]))
    
    logger.info('--------------------- Server Stage ---------------------')

    # Call server's main training/aggregation function
    # >>>>>>>>>>>>>>>>>> FedSDE Specific Code: Server Aggregation & Training <<<<<<<<<<<<<<<<<<
    if args.client_instance == 'FedSDE':
        logger.info("Server: Aggregating client knowledge and constructing global model (FedSDE)...")
        # For FedSDE, the server aggregates and trains using the collected soft labels
        # Pass the collected soft labels to the server's train method
        server.uploaded_soft_labels_dict = uploaded_soft_labels_dict 
        server.generated_data_diffusion=hard_sample_loader# Set this property on the server instance
        server.train() # This method will now use uploaded_soft_labels_dict
    # <<<<<<<<<<<<<<<<<< FedSDE Specific Code: Server Aggregation & Training >>>>>>>>>>>>>>>>>>>>
    else: # For other baselines, call their respective server.train()
        server.train()


if __name__ == "__main__":
    fed_run()