import torch
import cv2
import numpy as np
from torchvision import transforms
from utils.model_utils import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image(image_path: str):
    """
    Загружает и подготавливает изображение для модели.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


def apply_cut_model(checkpoint_path: str, input_image_path: str, output_image_path: str):
    """
    Загружает модель, применяет её к изображению и сохраняет результат.
    """
    # 1. Загружаем модель
    model = load_model(checkpoint_path, DEVICE)

    # 2. Предобрабатываем входное изображение
    input_tensor = preprocess_image(input_image_path)

    # 3. Применяем модель
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 4. Преобразуем в изображение и сохраняем
    output_image = postprocess_image(output_tensor)
    cv2.imwrite(output_image_path, output_image)

    print(f"✅ Обработанное изображение сохранено: {output_image_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Применение обученной модели CUT к изображению.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к файлу чекпоинта модели.")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному изображению.")
    parser.add_argument("--output", type=str, required=True, help="Путь к сохранённому выходному изображению.")
    args = parser.parse_args()

    apply_cut_model(args.checkpoint, args.input, args.output)
