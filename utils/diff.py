import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # For noise schedule

class ClassifierModel(nn.Module):
    """
    A simple Convolutional Neural Network Classifier.
    This serves as a placeholder. You can replace it with more complex
    architectures like LeNet, ResNet, etc., based on your 'args.sys_model'.
    """
    def __init__(self, img_channels, num_classes):
        super(ClassifierModel, self).__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(img_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, img_channels, 64, 64) # Assume 32x32 as common size for CIFAR
            dummy_output = self.feature_extractor(dummy_input)
            self.num_features = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.fc1 = nn.Linear(self.num_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No softmax here, CrossEntropyLoss expects logits
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-
                      np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                      ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# --- Helper Modules for U-Net ---

class Block(nn.Module):
    """
    A basic convolutional block with GroupNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))

class ResBlock(nn.Module):
    """
    A Residual Block that also takes timestep and optional class embeddings.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim=None, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) # Project time embedding to block's channels
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

        self.class_mlp = None
        if class_emb_dim is not None:
            self.class_mlp = nn.Linear(class_emb_dim, out_channels) # Project class embedding

    def forward(self, x, t_emb, c_emb=None):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        h += self.time_mlp(t_emb)[:, :, None, None] 

        # Add class embedding if provided
        if c_emb is not None and self.class_mlp is not None:
            # c_emb is (batch_size, class_emb_dim)
            h += self.class_mlp(c_emb)[:, :, None, None]

        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.skip_connection(x) # Add residual connection

class AttentionBlock(nn.Module):
    """
    Self-Attention block.
    """
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        batch, channels, H, W = x.shape
        h = self.norm(x)
        # Project and reshape
        q = self.query(h).reshape(batch, self.num_heads, channels // self.num_heads, H * W)
        k = self.key(h).reshape(batch, self.num_heads, channels // self.num_heads, H * W)
        v = self.value(h).reshape(batch, self.num_heads, channels // self.num_heads, H * W)

        # Compute attention scores
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.einsum('bncd,bnce->bnde', q, k) * scale  # (batch, num_heads, H*W, H*W)
        attn = attn.softmax(dim=-1)

        # Apply attention to v
        out = torch.einsum('bnde,bnce->bncd', attn, v)  # (batch, num_heads, channels_per_head, H*W)
        out = out.reshape(batch, channels, H, W)
        out = self.proj_out(out)
        return x + out  # Residual connection
    

class Downsample(nn.Module):
    """
    Downsampling block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """
    Upsampling block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1) # Using ConvTranspose2d for upsampling

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    The U-Net model for predicting noise in a Diffusion Model.
    This class now encapsulates the U-Net architecture.
    """
    def __init__(self, img_channels, base_channels, channel_multipliers, time_embedding_dim, class_embedding_dim=None):
        super().__init__()

        self.time_embedding_dim = time_embedding_dim
        self.class_embedding_dim = class_embedding_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        # Calculate channels for each level
        channels = [base_channels * mult for mult in channel_multipliers]
        
        # Downsampling path
        self.downs = nn.ModuleList()
        self.attns_down = nn.ModuleList()
        # Create pairs of (input_channel, output_channel) for down blocks
        in_out_channels = list(zip(channels[:-1], channels[1:]))
        # for i, (in_c, out_c) in enumerate(in_out_channels):
        #     self.downs.append(nn.ModuleList([
        #         ResBlock(in_c, out_c, self.time_embedding_dim, self.class_embedding_dim),
        #         ResBlock(out_c, out_c, self.time_embedding_dim, self.class_embedding_dim),
        #         Downsample(out_c) if i < len(in_out_channels) - 1 else nn.Identity() # Don't downsample after last block
        #     ]))
        #     # Add attention at certain resolutions (e.g., deeper levels)
        #     # This logic adds attention to the two deepest levels in the downsampling path
        #     if i >= len(in_out_channels) - 2: 
        #         self.attns_down.append(AttentionBlock(out_c))
        #     else:
        #         self.attns_down.append(nn.Identity()) # Placeholder for no attention
        self.skip_channels = []
        for i, (in_c, out_c) in enumerate(in_out_channels):
            self.downs.append(nn.ModuleList([
                ResBlock(in_c, out_c, self.time_embedding_dim, self.class_embedding_dim),
                ResBlock(out_c, out_c, self.time_embedding_dim, self.class_embedding_dim),
                Downsample(out_c) if i < len(in_out_channels) - 1 else nn.Identity()
            ]))
            if i >= len(in_out_channels) - 2: 
                self.attns_down.append(AttentionBlock(out_c))
            else:
                self.attns_down.append(nn.Identity())
            self.skip_channels.append(out_c)
        # Bottleneck
        self.mid_block1 = ResBlock(channels[-1], channels[-1], self.time_embedding_dim, self.class_embedding_dim)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], self.time_embedding_dim, self.class_embedding_dim)

        self.ups = nn.ModuleList()
        self.attns_up = nn.ModuleList()
        prev_channels = channels[-1]  # bottleneck输出
        for i in reversed(range(len(self.skip_channels))):
            skip_c = self.skip_channels[i]
            self.ups.append(nn.ModuleList([
                ResBlock(prev_channels + skip_c, skip_c, self.time_embedding_dim, self.class_embedding_dim),
                ResBlock(skip_c, skip_c, self.time_embedding_dim, self.class_embedding_dim),
                Upsample(skip_c) if i > 0 else nn.Identity()
            ]))
            self.attns_up.append(AttentionBlock(skip_c) if i < 2 else nn.Identity())
            prev_channels = skip_c  
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, self.skip_channels[0]),  # 改为 upsampling 最后一层输出通道
            nn.ReLU(),
            nn.Conv2d(self.skip_channels[0], img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t_emb, c_emb=None):
        """
        Forward pass of the U-Net.
        :param x: Noisy image tensor (B, C, H, W).
        :param t_emb: Time embedding tensor (B, time_embedding_dim).
        :param c_emb: Optional class embedding tensor (B, class_embedding_dim).
        :return: Predicted noise tensor (B, C, H, W).
        """
        x = self.conv_in(x)

        skips = []
        # Downsampling path
        for i, (resblock1, resblock2, downsample) in enumerate(self.downs):
            x = resblock1(x, t_emb, c_emb)
            x = resblock2(x, t_emb, c_emb)
            x = self.attns_down[i](x) # Apply attention if it's not nn.Identity
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb, c_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb, c_emb)

        # Upsampling path
        for i, (resblock1, resblock2, upsample) in enumerate(self.ups):
            skip = skips.pop()
            #print(f"Upsample stage {i}: x shape {x.shape}, skip shape {skip.shape}")
    
            x = torch.cat([x, skip], dim=1) # Concatenate with skip connection
            x = resblock1(x, t_emb, c_emb)
            x = resblock2(x, t_emb, c_emb)
            x = self.attns_up[i](x) # Apply attention if it's not nn.Identity
            x = upsample(x)

        noise_pred = self.conv_out(x)
        return noise_pred
class DiffusionModel(nn.Module):
    """
    A Denoising Diffusion Probabilistic Model (DDPM).
    This class manages the diffusion process, noise schedule, and delegates
    noise prediction to the UNet module.
    """
    def __init__(self, img_channels, img_size, num_timesteps=1000, num_classes=None, base_channels=64, channel_multipliers=[1, 2, 4, 8]):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.noise_steps = num_timesteps  # Number of diffusion steps
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes # For class-conditional generation

        # Timestep embedding layer for DiffusionModel
        self.time_embedding_dim = base_channels * 4 
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        )

        # Class embedding layer for DiffusionModel
        self.label_embedding = None
        self.class_embedding_dim = None
        if self.num_classes is not None:
            self.class_embedding_dim = base_channels * 4
            self.label_embedding = nn.Embedding(self.num_classes, self.class_embedding_dim)

        # Instantiate the UNet model - this is the key change!
        self.unet = UNet(
            img_channels=img_channels, 
            base_channels=base_channels, 
            channel_multipliers=channel_multipliers, 
            time_embedding_dim=self.time_embedding_dim,
            class_embedding_dim=self.class_embedding_dim if self.num_classes is not None else None
        )

        # 2. Define Noise Schedule (beta, alpha, alpha_bar) - Kept unchanged
        self.betas = self._cosine_noise_schedule(num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) 

        # Pre-compute terms needed for sampling/training - Kept unchanged
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        
        # Terms for reverse process (mean and variance of q(x_{t-1}|x_t, x_0)) - Kept unchanged
        self.posterior_variance = self.betas * (1.0 - self.alpha_bars.roll(1, dims=0) / (1.0 - self.alpha_bars))
        self.posterior_variance[0] = self.betas[0] 

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bars.roll(1, dims=0)) / (1.0 - self.alpha_bars)
        self.posterior_mean_coef2 = (1.0 - self.alphas) * torch.sqrt(1.0 - self.alpha_bars.roll(1, dims=0)) / (1.0 - self.alpha_bars)

    def sample_timesteps(self, n):
        # This function generates random timesteps for training batches - Kept unchanged
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def _cosine_noise_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule as in https://arxiv.org/abs/2102.09672"""
        t = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float64) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


    def forward(self, x, t, labels=None):
        """
        The forward pass of the DiffusionModel.
        It generates embeddings and passes them to the UNet to predict noise.
        
        :param x: Noisy image tensor.
        :param t: Timestep tensor.
        :param labels: Optional class label tensor.
        :return: Predicted noise tensor.
        """
        # MODIFIED: First call timestep_embedding function to get the initial embedding
        t_emb_initial = timestep_embedding(t, self.time_embedding_dim)
        # Then pass it through the time_mlp (which is now just linear layers)
        t_emb = self.time_mlp(t_emb_initial)
        
        # Generate class embedding using DiffusionModel's label_embedding
        c_emb = None
        if labels is not None and self.label_embedding is not None:
            if labels.dim() > 1: # If one-hot, convert to integer indices
                labels = torch.argmax(labels, dim=1)
            c_emb = self.label_embedding(labels)
        
        # Delegate noise prediction to the UNet instance
        noise_pred = self.unet(x, t_emb, c_emb)
        return noise_pred

    @torch.no_grad()
    def sample(self, num_samples, class_conditions=None, classifier=None, guidance_scale=0.0, device='cuda'):
        """
        Generate samples from the diffusion model using the reverse process.
        
        :param num_samples: Number of samples to generate.
        :param class_conditions: Optional tensor of one-hot class conditions (shape: num_samples, num_classes).
                                 If None, generates unconditionally.
        :param classifier: An optional ClassifierModel instance for classifier guidance.
        :param guidance_scale: Strength of classifier guidance.
        :param device: Device to perform sampling on.
        :return: Generated image tensor.
        """
        # Start with random noise (x_T ~ N(0, I))
        x_t = torch.randn(num_samples, self.img_channels, self.img_size, self.img_size, device=device)
        
        if class_conditions is not None:
            class_conditions = class_conditions.to(device)
            # DiffusionModel's forward method will handle conversion to integer indices if needed
            labels_for_diffusion_forward = class_conditions 
        else:
            labels_for_diffusion_forward = None

        # Iterate backwards through timesteps (denoising process)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

            # Predict the noise using the diffusion model's forward method (which in turn calls UNet)
            predicted_noise = self.forward(x_t, t_tensor, labels=labels_for_diffusion_forward)
            
            # --- Classifier Guidance (if enabled) ---
            if classifier is not None and guidance_scale > 0:
                x_t.requires_grad_(True)
                classifier.to(x_t.device)
                
                classifier_logits = classifier(x_t)
                log_probs = F.log_softmax(classifier_logits, dim=1)
                
                if class_conditions is not None:
                    target_class_indices = torch.argmax(class_conditions, dim=1)
                    guidance_loss = -F.nll_loss(log_probs, target_class_indices, reduction='sum')
                    
                    guidance_grad = torch.autograd.grad(guidance_loss, x_t)[0]
                    
                    predicted_noise = predicted_noise - guidance_scale * guidance_grad 
                
                x_t.requires_grad_(False)

            # --- Reverse Diffusion Step ---
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
            
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0) 

            posterior_mean_coef1_t = self.posterior_mean_coef1[t]
            posterior_mean_coef2_t = self.posterior_mean_coef2[t]
            mu_t_tilde = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t

            posterior_variance_t = self.posterior_variance[t]
            
            if t > 0:
                noise = torch.randn_like(x_t) 
                x_t = mu_t_tilde + torch.sqrt(posterior_variance_t) * noise
            else:
                x_t = mu_t_tilde 
        
        return x_t