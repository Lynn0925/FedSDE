from utils.fed_utils import assign_dataset, init_model
import torch
import torch.nn as nn

from utils.fed_utils import assign_dataset, init_model
from utils.util import Ensemble_A
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset # Added ConcatDataset
import logging

from .server_base import EnsembleServer # Assuming EnsembleServer is the base server class


logger = logging.getLogger(__name__)

class FedSDEServer(EnsembleServer):
    def __init__(self, args, client_ids, dataset_id, model_name, global_public_data_loader):
        
        super().__init__(args, client_ids, dataset_id, model_name)
        
        # Server Properties
        self.client_model = {}
        self.global_public_dataloader = global_public_data_loader
        self._id = "server"
        self._dataset_id = dataset_id
        self._model_name = model_name
        self._i_seed = args.sys_i_seed
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel, im_size=self._image_dim)
        
        self._epoch = args.server_n_epoch
        self._batch_size = args.server_bs
        self._lr = args.server_lr
        self._momentum = args.server_momentum
        self._num_workers = args.server_n_worker
        self.optim_name = args.server_optimizer
        self.uploaded_soft_labels_dict = {}
        self.combined_generated_data_loader = None # New attribute to hold combined generated data

        self.temperature = args.sde_temp  # Temperature for Knowledge Distillation
        self.kl_beta = args.kl_beta  # Weight for auxiliary CE loss on public data
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self._lr) # Using client LR for server, adjust as needed

        self.criterion_kl = nn.KLDivLoss(reduction='batchmean') # For KD loss (T^2 scaling applied manually)
        self.criterion_ce = nn.CrossEntropyLoss() # For auxiliary CE loss on public data

    def train(self):
        self.model.to(self._device)
        if not self.uploaded_soft_labels_dict:
            logger.warning("No soft labels uploaded by clients. Skipping global model training.")
            return

        use_generated_data = False
        if self.combined_generated_data_loader is not None and len(self.combined_generated_data_loader.dataset) > 0:
            use_generated_data = True
            logger.info("Server: Incorporating generated data from clients for training.")
        else:
            logger.warning("No valid generated data provided from clients. Training global model only on public data.")

        logger.info("Server: Aggregating client soft labels for public data...")
        
        # 1. Process Global Public Data
        all_public_images_list = []
        all_public_hard_labels_list = []
        for data, targets in self.global_public_dataloader:
            all_public_images_list.append(data)
            all_public_hard_labels_list.append(targets)
        public_images_tensor = torch.cat(all_public_images_list, dim=0).to(self._device)
        public_hard_labels_tensor = torch.cat(all_public_hard_labels_list, dim=0).to(self._device)

        num_classes = list(self.uploaded_soft_labels_dict.values())[0].shape[1]
        public_data_samples_count = public_images_tensor.shape[0]

        aggregated_soft_labels = torch.zeros(public_data_samples_count, num_classes).to(self._device)
        num_participating_clients = 0
        for client_id, soft_labels in self.uploaded_soft_labels_dict.items():
            if soft_labels is not None and soft_labels.numel() > 0:
                aggregated_soft_labels += soft_labels.to(self._device)
                num_participating_clients += 1
            else:
                logger.warning(f"Client {client_id} uploaded empty or None soft labels.")

        if num_participating_clients == 0:
            logger.error("No clients participated or uploaded valid soft labels. Cannot aggregate.")
            return

        aggregated_soft_labels /= num_participating_clients
        logger.info(f"Aggregated soft labels for public data from {num_participating_clients} clients. Shape: {aggregated_soft_labels.shape}")


        # 2. Prepare for Combined Dataset
        combined_images = public_images_tensor
        combined_teacher_soft_labels = aggregated_soft_labels
        combined_hard_labels = public_hard_labels_tensor

        if use_generated_data:
            all_generated_images_list = []
            all_generated_soft_labels_list = []
            all_generated_pseudo_labels_list = []

            for batch_idx, (gen_data, gen_soft_labels, gen_pseudo_labels) in enumerate(self.combined_generated_data_loader):
                all_generated_images_list.append(gen_data)
                all_generated_soft_labels_list.append(gen_soft_labels)
                all_generated_pseudo_labels_list.append(gen_pseudo_labels)

            if len(all_generated_images_list) > 0:
                generated_images_tensor = torch.cat(all_generated_images_list, dim=0).to(self._device)
                generated_soft_labels_tensor = torch.cat(all_generated_soft_labels_list, dim=0).to(self._device)
                generated_pseudo_labels_tensor = torch.cat(all_generated_pseudo_labels_list, dim=0).to(self._device)
                logger.info(f"Collected {generated_images_tensor.shape[0]} generated samples.")

                # Concatenate public and generated data
                combined_images = torch.cat((combined_images, generated_images_tensor), dim=0)
                combined_teacher_soft_labels = torch.cat((combined_teacher_soft_labels, generated_soft_labels_tensor), dim=0)
                combined_hard_labels = torch.cat((combined_hard_labels, generated_pseudo_labels_tensor), dim=0)
            else:
                logger.warning("No actual generated samples found in the combined generated data loader.")

        # Create a combined dataset for KD training
        kd_dataset = TensorDataset(combined_images.cpu(), combined_teacher_soft_labels.cpu(), combined_hard_labels.cpu())
        kd_dataloader = DataLoader(kd_dataset, batch_size=self._batch_size, shuffle=True)

        logger.info("Server: Training global model via Knowledge Distillation...")
        self.model.train()
        for epoch in range(self._epoch): # global_rounds is usually 1 for one-shot
            total_loss = 0
            num_batches = 0
            for batch_idx, (images, teacher_soft_labels, hard_labels) in enumerate(kd_dataloader):
                images, teacher_soft_labels, hard_labels = images.to(self._device), teacher_soft_labels.to(self._device), hard_labels.to(self._device)

                student_logits = self.model(images)
                student_soft_probs = F.log_softmax(student_logits / self.temperature, dim=1)

                # KD loss
                loss_kd = self.criterion_kl(student_soft_probs, teacher_soft_labels) * (self.temperature ** 2)

                # Auxiliary Cross-Entropy loss using hard labels (original for public, pseudo for generated)
                loss_ce = self.criterion_ce(student_logits, hard_labels)

                # Total loss for global model
                total_batch_loss = loss_kd + self.kl_beta * loss_ce # Use self.kl_beta for weighting
                
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                self.optimizer.step()

                total_loss += total_batch_loss.item()
                num_batches += 1

            avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Server KD Epoch {epoch+1}/{self._epoch}, Avg Loss: {avg_epoch_loss:.4f}")

        logger.info("Server: Global model training via Knowledge Distillation complete.")
        
        # After training, evaluate the global model
        accuracy, loss = self.evaluate_global_model()
        logger.info(f"Global Model Final Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    def evaluate_global_model(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        test_loader = DataLoader(self.testset, batch_size=200, shuffle=False) # Use self.testset for evaluation
        
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader.dataset):
                b_x = x.to(self._device)
                b_y = y.to(self._device)

                test_output = self.model(b_x)
                
                loss = F.cross_entropy(test_output, b_y, reduction='sum') # Sum loss for batch
                total_loss += loss.item()
                
                pred_y = torch.max(test_output, 1)[1]
                correct += (pred_y == b_y).sum().item()
                total += b_y.size(0)
                
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        logger.info(f"Evaluation on Testset - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        return accuracy, avg_loss
    def _get_model(self, model_name, dataset_id):
        pass 