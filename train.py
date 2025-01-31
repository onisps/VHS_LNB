import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.dataset import get_dataloader
from utils.cut_model import get_cut_model

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

def train_model(source_dir: str, target_dir: str):
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

        print(f"📉 Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, is_best=True)

    writer.close()
    print("🎉 Обучение завершено!")

if __name__ == "__main__":
    source = './data/patches/raw/'
    target = './data/patches/GT/'
    train_model(source, target)