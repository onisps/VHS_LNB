import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    """
    Остаточный блок для генератора Cycle-GAN.
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    Генератор Cycle-GAN (G_AB или G_BA).
    """
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()

        # Входной блок
        self.model = [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Свёрточные слои (downsampling)
        in_features = 64
        out_features = 128
        for _ in range(2):
            self.model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2

        # Остаточные блоки
        for _ in range(num_residual_blocks):
            self.model += [ResidualBlock(in_features)]

        # Декодер (upsampling)
        out_features = in_features // 2
        for _ in range(2):
            self.model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features //= 2

        # Выходной слой
        self.model += [
            nn.Conv2d(in_features, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    Дискриминатор PatchGAN для Cycle-GAN.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


def get_cycle_gan_model():
    """
    Создаёт Cycle-GAN с генераторами и дискриминаторами.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    G_AB = Generator().to(device)  # Преобразование A → B
    G_BA = Generator().to(device)  # Преобразование B → A
    D_A = Discriminator().to(device)  # Дискриминатор для домена A
    D_B = Discriminator().to(device)  # Дискриминатор для домена B

    # Оптимизаторы
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Функции потерь
    adversarial_loss = nn.MSELoss()  # Потеря для GAN
    cycle_loss = nn.L1Loss()  # Потеря согласованности цикла
    identity_loss = nn.L1Loss()  # Identity loss

    return G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, adversarial_loss, cycle_loss, identity_loss
