import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.dataset import get_dataloader
from utils.cycle_gan_model import get_cycle_gan_model
from utils.image_processing import merge_patches
from utils.global_variables import PATCH_SIZE
import cv2

# Параметры обучения
EPOCHS = 50
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"

# Готовим директории
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

def save_checkpoint(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, epoch):
    """
    Сохраняет чекпоинты модели Cycle-GAN.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cycle_gan_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D_A": optimizer_D_A.state_dict(),
        "optimizer_D_B": optimizer_D_B.state_dict(),
    }, checkpoint_path)
    print(f"✅ Чекпоинт сохранён: {checkpoint_path}")

def train_epoch(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, adversarial_loss, cycle_loss, identity_loss):
    """
    Один цикл обучения Cycle-GAN.
    """
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    epoch_G_loss = 0.0
    epoch_D_loss = 0.0

    for batch_idx, (real_A, real_B) in enumerate(dataloader):
        real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

        # Генерация изображений
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # Восстановленные изображения (цикл)
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)

        # -------------------------------
        # Обновление генераторов
        # -------------------------------
        optimizer_G.zero_grad()

        # Adversarial Loss (G_AB, G_BA должны обмануть дискриминаторы)
        loss_G_AB = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        loss_G_BA = adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))

        # Cycle Consistency Loss (должны восстановить оригиналы)
        loss_cycle_A = cycle_loss(rec_A, real_A)
        loss_cycle_B = cycle_loss(rec_B, real_B)

        # Identity Loss (если подать A в G_BA, он должен вернуть A)
        loss_identity_A = identity_loss(G_BA(real_A), real_A)
        loss_identity_B = identity_loss(G_AB(real_B), real_B)

        # Общий лосс генераторов
        loss_G = loss_G_AB + loss_G_BA + 10 * (loss_cycle_A + loss_cycle_B) + 5 * (loss_identity_A + loss_identity_B)
        loss_G.backward()
        optimizer_G.step()

        # -------------------------------
        # Обновление дискриминаторов
        # -------------------------------
        optimizer_D_A.zero_grad()
        loss_D_A_real = adversarial_loss(D_A(real_A), torch.ones_like(D_A(real_A)))
        loss_D_A_fake = adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_D_B_real = adversarial_loss(D_B(real_B), torch.ones_like(D_B(real_B)))
        loss_D_B_fake = adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        epoch_G_loss += loss_G.item()
        epoch_D_loss += (loss_D_A.item() + loss_D_B.item())

    return epoch_G_loss / len(dataloader), epoch_D_loss / len(dataloader)

def train_model(source_dir: str, target_dir: str, input_test_dir: str, output_test_dir: str, output_merged_dir: str):
    """
    Основной цикл обучения Cycle-GAN.
    """
    train_dataloader = get_dataloader(source_dir, target_dir, BATCH_SIZE, PATCH_SIZE)
    
    G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, adversarial_loss, cycle_loss, identity_loss = get_cycle_gan_model()
    G_AB.to(DEVICE)
    G_BA.to(DEVICE)
    D_A.to(DEVICE)
    D_B.to(DEVICE)

    best_G_loss = float("inf")
    epoch = 0
    train
    while train_G_loss < 0.2:
        train_G_loss, train_D_loss = train_epoch(
            G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, train_dataloader, adversarial_loss, cycle_loss, identity_loss
        )

        print(f"📉 Epoch {epoch}, G Loss: {train_G_loss:.4f}, D Loss: {train_D_loss:.4f}")
        writer.add_scalar("Loss/G_train", train_G_loss, epoch)
        writer.add_scalar("Loss/D_train", train_D_loss, epoch)

        if train_G_loss < best_G_loss:
            best_G_loss = train_G_loss
            save_checkpoint(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, epoch)
        apply_gan_model(
                        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cycle_gan_epoch_{epoch}.pth"),
                        input_test_dir,
                        output_test_dir,
                        generator_type="G_AB"
                      )
        image = cv2.imread('./data/Raw/raw/003_0009.jpg')
        original_size = image.shape[:2]
        merge_patches(output_test_dir, f'{output_merged_dir}', epoch, original_size)
        epoch += 1
    writer.close()
    print("🎉 Обучение завершено!")

if __name__ == "__main__":
    source = './data/patches/raw/'
    target = './data/patches/GT/'
    input_test_dir = './data/patches/test'
    output_test_dir = './data/patches/test_model'
    output_merged_dir = './data/test_model'

    os.makedirs(output_test_dir, exist_ok=True)
    os.makedirs(output_merged_dir, exist_ok=True)

    train_model(source, target, input_test_dir, output_test_dir, output_merged_dir)
