
from utils.fed_utils import assign_dataset, init_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import copy
import logging
from utils.diff import DiffusionModel
from .client_base import EnsembleClient

logger = logging.getLogger(__name__)

class FedSDEClient(EnsembleClient):
    def __init__(self, args, client_id, epoch, dataset_id, model_name,
                 private_data_loader, global_public_data_loader,
                 img_channels, img_size, num_classes):
        
        super().__init__(args, client_id, epoch, dataset_id, model_name)

        self.private_dataloader = private_data_loader
        self.global_public_dataloader = global_public_data_loader

        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.client_id = client_id

        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel, im_size=self._image_dim)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        self.diffusion_model = DiffusionModel(img_channels=self.img_channels, img_size=self.img_size).to(self._device)
        self.diffusion_optimizer = optim.Adam(self.diffusion_model.parameters(), lr=self.args.diff_lr)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def _train_diffusion_model(self):
        self.diffusion_model.train()
        logger.info(f"Client {self.client_id}: Training diffusion model for {self.args.diff_epochs} epochs...")
        for epoch in range(self.args.diff_epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_idx, (data, _) in enumerate(self.private_dataloader):
                data = data.to(self._device)
                
                timesteps = torch.randint(0, 1000, (data.shape[0],), device=self._device).long()
                output_from_diffusion = self.diffusion_model(data, timesteps) 
                loss = F.mse_loss(output_from_diffusion, data)

                self.diffusion_optimizer.zero_grad()
                loss.backward()
                self.diffusion_optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Client {self.client_id} Diff Epoch {epoch+1}/{self.args.diff_epochs}, Avg Loss: {avg_loss:.4f}")
        logger.info(f"Client {self.client_id}: Diffusion model training complete.")

    def _generate_hard_samples(self):
        self.diffusion_model.eval()
        self.model.eval()
        logger.info(f"Client {self.client_id}: Generating {self.args.num_hard_samples} hard samples...")
        
        with torch.no_grad():
            generated_images = self.diffusion_model.sample(
                num_samples=self.args.num_hard_samples,
                classifier=self.model,
                guidance_scale=1.0,
                device=self._device
            )
            
            if torch.isnan(generated_images).any() or torch.isinf(generated_images).any():
                logger.warning("Generated images contain NaN or Inf values. This might indicate an issue with diffusion model.")
                generated_images = torch.nan_to_num(generated_images, nan=0.0, posinf=1.0, neginf=-1.0)

            hard_sample_logits = self.model(generated_images)
            hard_sample_soft_labels = F.softmax(hard_sample_logits / self.args.temperature, dim=1)
            hard_sample_pseudo_labels = torch.argmax(hard_sample_logits, dim=1)
            
        logger.info(f"Client {self.client_id}: Hard sample generation complete. Generated {generated_images.shape[0]} samples.")
        return generated_images.detach(), hard_sample_soft_labels.detach(), hard_sample_pseudo_labels.detach()


    def _perform_local_self_distillation(self, generated_hard_samples, hard_sample_soft_labels, hard_sample_pseudo_labels):
        teacher_classifier = copy.deepcopy(self.model)
        teacher_classifier.eval()

        self.model.train()
        logger.info(f"Client {self.client_id}: Performing local self-distillation for {self.args.local_epochs} epochs...")

        private_images_list = []
        private_targets_list = []
        for data, targets in self.private_dataloader:
            private_images_list.append(data.cpu())
            private_targets_list.append(targets.cpu())
        
        private_images_tensor = torch.cat(private_images_list, dim=0)
        private_targets_tensor = torch.cat(private_targets_list, dim=0)
        
        with torch.no_grad():
            teacher_logits_private = teacher_classifier(private_images_tensor.to(self._device))
            private_soft_labels_from_teacher = F.softmax(teacher_logits_private / self.args.temperature, dim=1).cpu()

        all_sd_images = torch.cat((private_images_tensor, generated_hard_samples.cpu()), dim=0)
        all_sd_teacher_soft_labels = torch.cat((private_soft_labels_from_teacher, hard_sample_soft_labels.cpu()), dim=0)
        all_sd_hard_labels = torch.cat((private_targets_tensor, hard_sample_pseudo_labels.cpu()), dim=0)

        sd_dataset = TensorDataset(all_sd_images, all_sd_teacher_soft_labels, all_sd_hard_labels)
        sd_dataloader = DataLoader(sd_dataset, batch_size=self.args.batch_size, shuffle=True)


        for epoch in range(self.args.local_epochs):
            total_epoch_loss = 0.0
            num_batches = 0
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (data, teacher_soft_labels, hard_labels) in enumerate(sd_dataloader):
                data, teacher_soft_labels, hard_labels = data.to(self._device), teacher_soft_labels.to(self._device), hard_labels.to(self._device)
                
                student_logits = self.model(data)
                
                loss_ce = self.criterion_ce(student_logits, hard_labels)

                loss_kl = self.criterion_kl(
                    F.log_softmax(student_logits / self.args.temperature, dim=1),
                    teacher_soft_labels
                ) * (self.args.temperature ** 2)

                total_loss = loss_ce + loss_kl
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                total_epoch_loss += total_loss.item()
                num_batches += 1

                pred_y = torch.max(student_logits, 1)[1]
                correct_predictions += (pred_y == hard_labels).sum().item()
                total_samples += hard_labels.size(0)

                if batch_idx % 100 == 0:
                    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0
                    logger.debug(f"Client {self.client_id} SD Epoch {epoch}/{self.args.local_epochs}, Batch {batch_idx}, Loss: {total_loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            
            with torch.no_grad():
                for teacher_param, student_param in zip(teacher_classifier.parameters(), self.model.parameters()):
                    teacher_param.data.mul_(0.999).add_(student_param.data, alpha=0.001)

            avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
            avg_epoch_accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0
            logger.info(f"Client {self.client_id} SD Epoch {epoch+1}/{self.args.local_epochs} complete. Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%")
        logger.info(f"Client {self.client_id}: Local self-distillation complete.")


    def _generate_soft_labels_for_public_data(self):
        self.model.eval()
        soft_labels_list = []
        with torch.no_grad():
            for data, _ in self.global_public_dataloader:
                data = data.to(self._device)
                logits = self.model(data)
                soft_probs = F.softmax(logits / self.args.temperature, dim=1)
                soft_labels_list.append(soft_probs.cpu())
        
        soft_labels_tensor = torch.cat(soft_labels_list, dim=0)
        logger.info(f"Client {self.client_id}: Generated soft labels for public data. Shape: {soft_labels_tensor.shape}")
        return soft_labels_tensor

    def train(self, model=None):
        if model is not None:
             self.set_model(model.state_dict())
        
        # 0. Train local classifier on private data
        self.model.train()
        self.model.to(self._device)
        logger.info(f"Client {self.client_id}: Training classifier on private data for {self.args.local_epochs} epochs...")
        for epoch in range(self.args.local_epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            num_batches = 0
            for batch_idx, (data, targets) in enumerate(self.private_dataloader):
                data, targets = data.to(self._device), targets.to(self._device)
                logits = self.model(data)
                loss = self.criterion_ce(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

                pred_y = torch.max(logits, 1)[1]
                correct_predictions += (pred_y == targets).sum().item()
                total_samples += targets.size(0)

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0
            logger.info(f"Client {self.client_id}: Local Classifier Epoch {epoch+1}/{self.args.local_epochs}, Avg Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

        # 1. Train local Diffusion Model
        self._train_diffusion_model()
        
        # 2. Generate hard samples using Diffusion Model and Classifier Guidance
        generated_hard_samples, hard_sample_soft_labels, hard_sample_pseudo_labels = self._generate_hard_samples()
        
        hard_sample_dataset = TensorDataset(
            generated_hard_samples.cpu(),
            hard_sample_soft_labels.cpu(), 
            hard_sample_pseudo_labels.cpu()
        )
        
        # 3. Perform Local Self-Distillation for Classifier using private data and generated hard samples
        self._perform_local_self_distillation(generated_hard_samples, hard_sample_soft_labels, hard_sample_pseudo_labels)

        # 4. Generate Soft Labels for Public Data using the refined classifier
        soft_labels_for_public_data = self._generate_soft_labels_for_public_data()

        return soft_labels_for_public_data, hard_sample_dataset
