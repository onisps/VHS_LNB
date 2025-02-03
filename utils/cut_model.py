import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# For Torchvision versions >=0.13, the weights parameter has been updated.
try:
    from torchvision.models import ResNet18_Weights
    pretrained_weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=pretrained_weights)
except ImportError:
    encoder = resnet18(weights="pretrained")


class ProperDecoder(nn.Module):
    """
    An adaptive decoder that upsamples from (N, latent_dim, 1, 1) to (N, 3, patch_size, patch_size).
    The decoder is constructed dynamically so that the number of transposed convolution blocks
    depends on the desired output resolution (patch_size). The first block increases the spatial dimensions
    from 1×1 to 4×4, and subsequent blocks double the resolution until the target size is reached.
    
    This adaptive strategy is similar in spirit to dynamic upsampling techniques 
    used in generative adversarial networks (Radford et al., 2015) and fully convolutional networks (Long et al., 2015).
    """

    def __init__(self, latent_dim=512, out_channels=3, patch_size=512):
        """
        Args:
            latent_dim (int): Dimensionality of the latent feature vector.
            out_channels (int): Number of output channels (typically 3 for RGB images).
            patch_size (int): Desired output resolution (assumed square). Must be a power-of-2 and at least 4.
        """
        super().__init__()

        if patch_size < 4 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError("patch_size must be a power of 2 and at least 4.")

        self.patch_size = patch_size

        # The first layer upsamples from 1x1 to 4x4.
        layers = []
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        ))
        current_channels = 512

        # Compute the number of doubling layers required.
        # Starting from 4, we double until we reach patch_size:
        # 4 * 2^(n) = patch_size  ->  n = log2(patch_size/4)
        doubling_count = int(math.log2(patch_size / 4))

        # Add doubling layers: each layer doubles the spatial dimensions.
        # Channel dimensions are reduced by half until a minimum of 8 channels.
        for i in range(1, doubling_count + 1):
            next_channels = max(512 // (2 ** i), 8)
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(True),
            ))
            current_channels = next_channels

        self.upsampling = nn.Sequential(*layers)
        # Final convolution to refine the output features to the desired number of channels.
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Suitable when output pixels are scaled to [-1, 1]
        )

    def forward(self, x):
        x = self.upsampling(x)
        x = self.output_conv(x)
        return x


class CUT(nn.Module):
    """
    A simple implementation of CUT (Contrastive Unpaired Translation) that is adaptive to the target patch size.
    
    The model consists of:
      1. A ResNet-18 encoder (with the final fully connected layer removed) to extract latent features.
      2. An adaptive decoder (ProperDecoder) that upsamples features to generate high-resolution images
         of size (patch_size x patch_size).
    
    This design is consistent with recent approaches in unpaired image-to-image translation 
    (Park et al., 2020) and generative modeling (Radford et al., 2015).
    """

    def __init__(self, patch_size=512):
        super(CUT, self).__init__()
        # Use the encoder defined above.
        if 'encoder' not in globals():
            self.encoder = resnet18(weights="pretrained")
        else:
            self.encoder = encoder
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer.

        # Instantiate the adaptive decoder.
        self.decoder = ProperDecoder(latent_dim=512, out_channels=3, patch_size=patch_size)

    def forward(self, x):
        # Encode: input x of shape (N, 3, H, W) -> latent features of shape (N, 512)
        features = self.encoder(x)
        # Reshape features for the decoder.
        features = features.view(features.size(0), 512, 1, 1)
        # Decode: upsample to (N, 3, patch_size, patch_size)
        output = self.decoder(features)
        return output


def get_cut_model(patch_size=512):
    """
    Loads the CUT model with the adaptive decoder, along with its optimizer and loss criterion.
    
    Args:
        patch_size (int): Desired output patch size (assumed square and power-of-2).
    
    Returns:
        model: The CUT model moved to CUDA (if available).
        optimizer: Adam optimizer with a learning rate of 0.0002 and betas (0.5, 0.999).
        criterion: L1 loss for reconstruction.
    """
    model = CUT(patch_size=patch_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    return model, optimizer, criterion
