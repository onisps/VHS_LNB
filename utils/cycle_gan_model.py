import torch
import torch.nn as nn
import torch.optim as optim
from utils.global_variable import LR
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    """
    Residual block using reflection padding + instance normalization,
    as recommended for CycleGAN-style networks.
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN Generator with reflection padding and instance normalization.
    By default, uses 9 residual blocks (typical for 256×256 or 512×512 images).
    """
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial block: reflection pad + 7×7 conv + instance norm + ReLU
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features //= 2

        # Final block: reflection pad + 7×7 conv + Tanh
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with instance normalization, following CycleGAN defaults.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4,
                                 stride=stride, padding=1, bias=False))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # The first block does not use normalization
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            # Last block with stride=1
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


def get_cycle_gan_model():
    """
    Creates and returns:
      - Generators G_AB, G_BA
      - Discriminators D_A, D_B
      - Their respective Adam optimizers
      - Common loss functions (adversarial, cycle, identity)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G_AB = Generator().to(device)  # A → B
    G_BA = Generator().to(device)  # B → A
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=LR, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

    # Loss functions
    adversarial_loss = nn.MSELoss()  # or nn.BCEWithLogitsLoss() in some variations
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    return G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, \
           adversarial_loss, cycle_loss, identity_loss
