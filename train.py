import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.dataset import get_dataloader
from utils.cut_model import get_cut_model
from apply_model import apply_cut_model
from utils.image_processing import merge_patches
from make_patches import main
from datetime import datetime
from glob2 import glob
import cv2

# Параметры обучения
EPOCHS = 50
BATCH_SIZE = 8
PATCH_SIZE = (512, 512)  # Updated for 512x512 images
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"

# Готовим логи и директории
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

def save_checkpoint(model, optimizer, epoch, is_best=False):
    """
    Сохраняет чекпоинт модели.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cut_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "cut_best.pth")
        torch.save(model.state_dict(), best_path)
    print(f"✅ Чекпоинт сохранён: {checkpoint_path}")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (source, target) in enumerate(dataloader):
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(source)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for source, target in dataloader:
            source, target = source.to(device), target.to(device)
            output = model(source)
            loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / len(dataloader)

def train_model(source_dir: str, target_dir: str, input_test_dir: str, output_test_dir: str, output_merged_dir: str):
    """
    Основной цикл обучения CUT.
    """
    train_dataloader = get_dataloader(source_dir, target_dir, BATCH_SIZE, PATCH_SIZE)
    val_dataloader = get_dataloader(source_dir, target_dir, BATCH_SIZE, PATCH_SIZE)
    model, optimizer, criterion = get_cut_model()
    model.to(DEVICE)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, DEVICE)
        val_loss = validate_model(model, val_dataloader, criterion, DEVICE)

        print(f"📉 Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, is_best=True)

        apply_cut_model(model, input_test_dir, output_test_dir)
        image = cv2.imread('./data/Raw/raw/003_0009.jpg')
        original_size = image.shape[:2]
        merge_patches(output_test_dir, f'{output_merged_dir}', epoch, original_size)
    writer.close()
    print("🎉 Обучение завершено!")

if __name__ == "__main__":

    os.makedirs('data/patches', exist_ok=True)
    os.makedirs('data/patches/raw', exist_ok=True)
    os.makedirs('data/patches/GT', exist_ok=True)
    os.makedirs('data/patches/test', exist_ok=True)
    now = datetime.now()
    output_patches_raw =  './data/patches/raw'       # "Папка для сохранения патчей."
    output_patches_gt =  './data/patches/GT'       # "Папка для сохранения патчей."
    output_patches_test =  './data/patches/test'       # "Папка для сохранения патчей."
    for file in glob('./data/Raw/raw/*'):
        image = file                    # "Путь к исходному изображению."
        main(image, output_patches_raw)
    print(f'Done with train raw patches | time > {datetime.now() - now}')
    now = datetime.now()
    for file in glob('./data/Raw/GT/*'):
        image = file                    # "Путь к исходному изображению."
        main(image, output_patches_gt)
    print(f'Done with raw GT patches | time > {datetime.now() - now}')
    now = datetime.now()
    for file in glob('./data/Raw/test/raw/*'):
        image = file                    # "Путь к исходному изображению."
        main(image, output_patches_test)
    print(f'Done with raw test patches | time > {datetime.now() - now}')

    source = './data/patches/raw/'
    target = './data/patches/GT/'
    input_test_dir = './data/patches/test'
    output_test_dir = './data/patches/test_model'
    output_merged_dir = './data/test_model'
    os.makedirs(output_test_dir, exist_ok=True)
    os.makedirs(output_merged_dir, exist_ok=True)
    train_model(source, target, input_test_dir, output_test_dir, output_merged_dir)