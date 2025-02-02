import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18


class ProperDecoder(nn.Module):
    """
    A decoder that upsamples from (N, latent_dim, 1, 1) to (N, 3, 512, 512).
    Uses ConvTranspose2d layers to progressively double spatial dimensions.
    """

    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()

        # 1) First upsampling block: (1x1) -> (4x4)
        #    kernel_size=4, stride=1, padding=0
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # 2) (4x4) -> (8x8)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # 3) (8x8) -> (16x16)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # 4) (16x16) -> (32x32)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # 5) (32x32) -> (64x64)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        # 6) (64x64) -> (128x128)
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        # 7) (128x128) -> (256x256)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        # 8) (256x256) -> (512x512)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        # Final output to 3 channels; optional Tanh or Sigmoid
        # Here we use a simple 3×3 conv to refine the last feature map
        self.output_conv = nn.Sequential(
            nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # or nn.Sigmoid(), or no activation if you prefer raw outputs
        )

    def forward(self, x):
        """
        x is expected to have shape (N, latent_dim, 1, 1).
        """
        x = self.up1(x)  # (N, 512, 4, 4)
        x = self.up2(x)  # (N, 256, 8, 8)
        x = self.up3(x)  # (N, 128, 16, 16)
        x = self.up4(x)  # (N, 64, 32, 32)
        x = self.up5(x)  # (N, 32, 64, 64)
        x = self.up6(x)  # (N, 16, 128, 128)
        x = self.up7(x)  # (N, 8, 256, 256)
        x = self.up8(x)  # (N, 8, 512, 512)
        x = self.output_conv(x)  # (N, 3, 512, 512)
        return x


class CUT(nn.Module):
    """
    Простая реализация CUT (Contrastive Unpaired Translation).
    """

    def __init__(self):
        super(CUT, self).__init__()
        self.encoder = resnet18(weights="pretrained")
        self.encoder.fc = nn.Identity()  # Убираем последний слой

        # 2) Decoder
        self.decoder = ProperDecoder(latent_dim=512, out_channels=3)

    def forward(self, x):
        # Encode: shape (N, 3, H, W) -> (N, 512) after resnet18 global pooling
        features = self.encoder(x)

        # Reshape for decoder
        features = features.view(features.size(0), 512, 1, 1)

        # Decode: shape (N, 512, 1, 1) -> (N, 3, 512, 512)
        output = self.decoder(features)
        return output


def get_cut_model():
    """
    Загружает CUT модель и оптимизатор.
    """
    model = CUT().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    return model, optimizer, criterion