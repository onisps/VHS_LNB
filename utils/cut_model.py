import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

class CUT(nn.Module):
    """
    Простая реализация CUT (Contrastive Unpaired Translation).
    """
    def __init__(self):
        super(CUT, self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # Убираем последний слой

        # Adjusted decoder for 128x128 output
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 2x2
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Output: 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output: 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # Output: 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),     # Output: 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),      # Output: 128x128
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)  # Reshape for decoder
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)  # Resize to 512x512
        return x

def get_cut_model():
    """
    Загружает CUT модель и оптимизатор.
    """
    model = CUT().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    return model, optimizer, criterion
