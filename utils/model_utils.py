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
    Загружает генератор Cycle-GAN из файла чекпоинта.
    
    :param checkpoint_path: Путь к чекпоинту модели.
    :param device: Устройство для загрузки модели (например, 'cuda' или 'cpu').
    :param generator_type: Выбор генератора ('G_AB' для преобразования A → B или 'G_BA' для B → A).
    :return: Загруженная модель генератора.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Instantiate the Generator (as defined in your cycle_gan_model.py)
    from utils.cycle_gan_model import Generator  # Импорт класса генератора
    model = Generator().to(device)
    
    # Load the corresponding generator state_dict from the checkpoint
    if generator_type.upper() == 'G_AB':
        if "G_AB" in checkpoint:
            model.load_state_dict(checkpoint["G_AB"])
        else:
            raise KeyError("Ключ 'G_AB' не найден в чекпоинте.")
    elif generator_type.upper() == 'G_BA':
        if "G_BA" in checkpoint:
            model.load_state_dict(checkpoint["G_BA"])
        else:
            raise KeyError("Ключ 'G_BA' не найден в чекпоинте.")
    else:
        raise ValueError("generator_type должен быть 'G_AB' или 'G_BA'.")
    
    model.eval()  # Переключаем модель в режим инференса
    print(f"✅ Генератор {generator_type} загружен из {checkpoint_path} (эпоха {checkpoint.get('epoch', 'неизвестна')})")
    return model

def load_model(checkpoint_path: str, device: str = "cuda", model_type: str = 'GAN', generator_type: str = 'G_AB'):
    if model_type.upper() == 'CUT':
        return load_model_cut(checkpoint_path, device)
    elif model_type.upper() == 'GAN':
        return load_model_gan(checkpoint_path, device, generator_type)
    else:
        raise ValueError("model_type должен быть 'CUT' или 'GAN'.")
