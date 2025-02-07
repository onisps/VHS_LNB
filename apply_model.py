import torch
import cv2
import numpy as np
from torchvision import transforms
from utils.model_utils import load_model
from utils.image_processing import merge_patches
import os
from utils.global_variables import PATCH_SIZE


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image(image_path: str):
    """
    Загружает и подготавливает изображение для модели.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    except:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    return transform(image).unsqueeze(0).to(DEVICE)


def postprocess_image(tensor):
    """
    Преобразует выход модели обратно в изображение.
    """
    tensor = tensor.cpu().detach().squeeze(0)
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # Обратно в [0,1]
    image_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def apply_cut_model(model, input_dir: str, output_dir: str):
    """
    Загружает модель, применяет её к изображению и сохраняет результат.
    """
    # 1. Загружаем модель
    # model = load_model(checkpoint_path, DEVICE)

    # 2. Обрабатываем все изображения в директории
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)

            # 3. Предобрабатываем входное изображение
            input_tensor = preprocess_image(input_image_path)

            # 4. Применяем модель
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # 5. Преобразуем в изображение и сохраняем
            output_image = postprocess_image(output_tensor)
            cv2.imwrite(output_image_path, output_image)

            # print(f"✅ Обработанное изображение сохранено: {output_image_path}")

def apply_gan_model(checkpoint_path: str, input_dir: str, output_dir: str, generator_type="G_AB"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем модель
    print(f"Загрузка модели из чекпоинта: {checkpoint_path}")
    model = load_model(checkpoint_path, device=DEVICE, generator_type=generator_type)
    if model is None:
        raise ValueError("Модель не была загружена.")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
    
            # Предобработка изображения
            print(f"Обработка изображения: {input_image_path}")
            input_tensor = preprocess_image(input_image_path)
    
            # Применение модели
            with torch.no_grad():
                output_tensor = model(input_tensor)
    
            # Преобразование в изображение и сохранение
            output_image = postprocess_image(output_tensor)
            cv2.imwrite(output_image_path, output_image)
            print(f"Изображение сохранено: {output_image_path}")
    
if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Применение обученной модели CUT к изображению.")
    # parser.add_argument("--checkpoint", type=str, required=True, help="Путь к файлу чекпоинта модели.")
    # parser.add_argument("--input", type=str, required=True, help="Путь к входному изображению.")
    # parser.add_argument("--output", type=str, required=True, help="Путь к сохранённому выходному изображению.")
    # args = parser.parse_args()
    # apply_cut_model(args.checkpoint, args.input, args.output)
    os.makedirs('data/patches/test_model', exist_ok=True)
    os.makedirs('data/test_model', exist_ok=True)
    input = './data/patches/test/'
    output = './data/patches/test_model/'
    epoch = 30
    checkpoint = f'./checkpoints/cut_epoch_{epoch}.pth'
    model = load_model(checkpoint, DEVICE)
    apply_cut_model(checkpoint, input, output)

    image = cv2.imread('./data/Raw/raw/003_0009.jpg')
    original_size = image.shape[:2]
    merge_patches(output, f'data/test_model/reconstructed_epoch_{epoch}.jpg', original_size)
    # merge_patches('/home/esh/PycharmProjects/VHS_LNB/data/Raw/test/raw/', 'data/test_model/raw_reconstructed_pic.jpg', original_size)
