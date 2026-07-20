To handle multiple image labels using a Convolutional Autoencoder (CAE), you must combine unsupervised reconstruction loss with supervised multi-label classification loss. This multi-task learning approach uses a single CNN encoder to compress the image into a bottleneck (latent space), which branches into two parallel paths: a decoder to reconstruct the image, and a classification head to predict multiple labels simultaneously

https://www.sciencedirect.com/science/article/pii/S2211379726000343

Core Structural ConceptsWhen building this architecture, you must adapt standard autoencoder layers to support classification:The Shared Encoder: Convolutional layers compress the spatial dimensions of the input image down to a low-dimensional latent vector (bottleneck). This forces the network to learn generalized, high-quality image features.

The Image Decoder Branch: Up-sampling and transposed convolutional layers reconstruct the original pixel map from the bottleneck vector.

https://www.youtube.com/watch?v=gOxBRgYUXaU

The Multi-Label Head Branch: A fully connected (dense) layer branches off the bottleneck vector. Because an image can possess multiple labels at once (e.g., a photo with both [cat, indoor, sitting]), this branch must use a Sigmoid activation function instead of Softmax. This creates an independent probability score between 0 and 1 for every individual label.

https://stackoverflow.com/questions/48493689/does-training-a-multi-labeling-cnns-on-single-class-data-hinder-accuracy

Step-by-Step Multi-Task Model Implementation

The following complete PyTorch implementation demonstrates how to build and train a multi-output Convolutional Autoencoder.

pythonimport torch
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


Dual Loss Optimization StrategyTo train this network successfully, you must compute two separate loss functions and optimize them together using a weighted balance variable (α):Reconstruction Loss (\(L_{recon}\)): Measures the pixel-by-pixel difference between the original and reconstructed image. Mean Squared Error (MSE) is typically used for continuous pixel data.Classification Loss (\(L_{class}\)): Evaluates label predictions. Because multiple labels can be true simultaneously, Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) must be used. It applies a Sigmoid activation to your logits internally for numerical stability.

\(Total\ Loss=(1-\alpha )\cdot L_{recon}+\alpha \cdot L_{class}\)

python# Initialize model, optimizer, and loss criteria
num_labels = 10  # Example count
model = MultiLabelCNNAutoencoder(num_classes=num_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define distinct loss components
criterion_recon = nn.MSELoss()
criterion_class = nn.BCEWithLogitsLoss() 

# Loss weighting hyperparameter (Balance factor between 0 and 1)
alpha = 0.5 

# Training Loop Stub
for images, labels in dataloader: # labels shape: [batch_size, num_labels] (One-hot/Multi-hot encoded)
    optimizer.zero_grad()
    
    # Forward Pass
    reconstructed, logits = model(images)
    
    # Calculate Losses
    loss_recon = criterion_recon(reconstructed, images)
    loss_class = criterion_class(logits, labels.float()) # Labels must be float for BCE
    
    # Combined Loss
    total_loss = ((1.0 - alpha) * loss_recon) + (alpha * loss_class)
    
    # Backward Pass
    total_loss.backward()
    optimizer.step()

Tips for Peak Performance

* Tune the Alpha (α) Hyperparameter: If your model reconstructs perfectly but cannot classify, increase α (e.g., to 0.7). If the classification is accurate but reconstructions are blurry, decrease α.

* Format Labels to Multi-Hot Encoding: Ensure your targets are binary arrays where every applicable class is marked with 1.0 and missing ones are 0.0 (e.g., [1.0, 0.0, 1.0, 1.0])
https://stackoverflow.com/questions/43865804/how-to-perform-multi-labeling-classification-for-cnn

* Set Thresholding for Predictions: During inference, pass your raw classification outputs through torch.sigmoid(logits). Since multiple labels can trigger, tag any label with a value surpassing a set threshold (typically > 0.5) as present. https://stackoverflow.com/questions/48493689/does-training-a-multi-labeling-cnns-on-single-class-data-hinder-accuracy
