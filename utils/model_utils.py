import os
import torch
from utils.cut_model import CUT

CHECKPOINT_DIR = "../checkpoints"


def save_model(model, optimizer, epoch):
    """
    Сохраняет модель в файле чекпоинта.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cut_model_epoch_{epoch}.pth")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

    print(f"✅ Модель сохранена: {checkpoint_path}")


def load_model(checkpoint_path: str, device: str = "cuda"):
    """
    Загружает модель CUT из файла чекпоинта.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CUT().to(device)

    # Handle missing keys
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("⚠️ 'model_state_dict' not found in checkpoint. Loading model without weights.")

    # model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Переключаем в режим инференса
    print(f"✅ Модель загружена из {checkpoint_path} (эпоха {checkpoint['epoch']})")
    return model
