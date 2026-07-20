import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelCNNAutoencoder(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelCNNAutoencoder, self).__init__()
        
        # 1. ENCODER: Compresses image (e.g., 3 channels, 64x64)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Out: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Out: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Out: 64 x 8 x 8
            nn.ReLU()
        )
        
        # Flattened bottleneck dimension: 64 * 8 * 8 = 4096
        self.bottleneck_dim = 64 * 8 * 8
        
        # 2. DECODER BRANCH: Reconstructs the original image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Out: 32 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Out: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Out: 3 x 64 x 64
            nn.Sigmoid() # Keeps pixel outputs bound safely between 0 and 1
        )
        
        # 3. MULTI-LABEL CLASSIFICATION BRANCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bottleneck_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # Outputs raw logits for each class
        )

    def forward(self, x):
        # Shared feature extraction
        latent_features = self.encoder(x)
        
        # Branch 1: Image Reconstruction
        reconstructed_image = self.decoder(latent_features)
        
        # Branch 2: Multi-Label Classification
        label_logits = self.classifier(latent_features)
        
        return reconstructed_image, label_logits
