import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.dataset import get_dataloader
from utils.cut_model import get_cut_model

# Параметры обучения
EPOCHS = 50
BATCH_SIZE = 8
PATCH_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"

# Готовим логи и директории
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

def save_checkpoint(model, optimizer, epoch):
    """
    Сохраняет чекпоинт модели.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cut_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"✅ Чекпоинт сохранён: {checkpoint_path}")

def train_model(source_dir: str, target_dir: str):
    """
    Основной цикл обучения CUT.
    """
    dataloader = get_dataloader(source_dir, target_dir, BATCH_SIZE, PATCH_SIZE)
    model, optimizer, criterion = get_cut_model()
    model.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Эпоха {epoch+1}/{EPOCHS}")
        for batch_idx, (source, target) in progress_bar:
            source, target = source.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(source)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + batch_idx)

        avg_loss = epoch_loss / len(dataloader)
        print(f"📉 Средний лосс: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch)

    writer.close()
    print("🎉 Обучение завершено!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение модели CUT.")
    parser.add_argument("--source", type=str, required=True, help="Путь к папке с немаркированными изображениями.")
    parser.add_argument("--target", type=str, required=True, help="Путь к папке с окрашенными изображениями.")
    args = parser.parse_args()

    train_model(args.source, args.target)
