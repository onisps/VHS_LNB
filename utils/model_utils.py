import os
import torch
from utils.cut_model import CUT
from utils.cycle_gan_model import Generator
import torch.nn as nn
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


def load_model_cut(checkpoint_path: str, device: str = "cuda"):
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
    model.eval()  # Переключаем в режим инференса
    print(f"✅ Модель загружена из {checkpoint_path} (эпоха {checkpoint['epoch']})")
    return model

    
def load_model_gan(checkpoint_path: str, device: str = "cuda", generator_type: str = 'G_AB'):
    """
    Loads a CycleGAN generator from a checkpoint, wrapping it in DataParallel
    if multiple GPUs are available. This allows you to load state_dict keys
    that contain the 'module.' prefix.

    :param checkpoint_path: Path to the checkpoint file.
    :param device: Device to load the model onto ('cuda' or 'cpu').
    :param generator_type: Which generator to load ('G_AB' or 'G_BA').
    :return: A loaded generator model, potentially wrapped in DataParallel.
    """
    # Load the checkpoint from disk
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Instantiate the Generator as defined in your cycle_gan_model.py
    model = Generator()  # do NOT move to device yet, we will wrap first

    # Wrap the model in DataParallel if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for loading!")
        model = nn.DataParallel(model)

    # Now move to the specified device (e.g., 'cuda:0')
    model.to(device)

    # Select which generator weights to load from the checkpoint
    if generator_type.upper() == 'G_AB':
        if "G_AB" in checkpoint:
            model.load_state_dict(checkpoint["G_AB"])
        else:
            raise KeyError("Key 'G_AB' not found in checkpoint.")
    elif generator_type.upper() == 'G_BA':
        if "G_BA" in checkpoint:
            model.load_state_dict(checkpoint["G_BA"])
        else:
            raise KeyError("Key 'G_BA' not found in checkpoint.")
    else:
        raise ValueError("generator_type must be 'G_AB' or 'G_BA'.")

    model.eval()
    print(f"✅ Generator {generator_type} loaded from {checkpoint_path} "
          f"(epoch {checkpoint.get('epoch', 'unknown')}) with DataParallel={num_gpus>1}")
    return model


def load_model(checkpoint_path: str, device: str = "cuda", model_type: str = 'GAN', generator_type: str = 'G_AB'):
    if model_type.upper() == 'CUT':
        return load_model_cut(checkpoint_path, device)
    elif model_type.upper() == 'GAN':
        return load_model_gan(checkpoint_path, device, generator_type)
    else:
        raise ValueError("model_type должен быть 'CUT' или 'GAN'.")
