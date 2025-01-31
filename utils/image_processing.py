import os
import cv2
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from utils.metadata import get_image_metadata

PATCH_SIZE = 512  # Размер патча (2048x2048)
OVERLAP = 128  # Перекрытие патчей (10-20%)


def create_patches(image_path: str, output_dir: str) -> List[str]:
    """
    Разрезает изображение на патчи с перекрытием и сохраняет их.

    :param image_path: Путь к исходному изображению.
    :param output_dir: Директория для сохранения патчей.
    :return: Список путей к сохранённым патчам.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    patch_paths = []
    patch_idx = 0

    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP):
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_filename = f"{output_dir}/{image_path.split('/')[-1].split('.')[0]}-patch_{patch_idx:04d}.png"
            cv2.imwrite(patch_filename, patch)
            patch_paths.append(patch_filename)
            patch_idx += 1

    return patch_paths


def merge_patches(patch_dir: str, output_image_path: str, original_size: Tuple[int, int]):
    """
    Обратно собирает изображение из патчей с учётом перекрытия.

    :param patch_dir: Директория с патчами.
    :param output_image_path: Путь для сохранения восстановленного изображения.
    :param original_size: Оригинальный размер изображения (width, height).
    """
    width, height = original_size
    stitched_image = np.zeros((height, width, 3), dtype=np.float32)  # Используем float для точного усреднения
    weight_map = np.zeros((height, width), dtype=np.float32)  # Карта весов для усреднения

    patch_files = sorted(os.listdir(patch_dir))  # Загружаем патчи в правильном порядке
    patch_idx = 0

    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP):
            patch_path = os.path.join(patch_dir, patch_files[patch_idx])
            patch = cv2.imread(patch_path).astype(np.float32)  # Загружаем патч как float
            patch_idx += 1

            # Создаём маску весов для текущего патча
            patch_weight = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

            # Уменьшаем вес на краях патча для сглаживания перекрытий
            if OVERLAP > 0:
                # Горизонтальные края
                patch_weight[:OVERLAP, :] *= np.linspace(0, 1, OVERLAP)[:, np.newaxis]
                patch_weight[-OVERLAP:, :] *= np.linspace(1, 0, OVERLAP)[:, np.newaxis]

                # Вертикальные края
                patch_weight[:, :OVERLAP] *= np.linspace(0, 1, OVERLAP)
                patch_weight[:, -OVERLAP:] *= np.linspace(1, 0, OVERLAP)

            # Добавляем патч и его веса к изображению
            stitched_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += patch * patch_weight[:, :, np.newaxis]
            weight_map[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += patch_weight

    # Усредняем значения, чтобы избежать резких границ
    weight_map[weight_map == 0] = 1  # Чтобы избежать деления на 0
    stitched_image = (stitched_image / weight_map[:, :, np.newaxis]).astype(np.uint8)

    # Сохраняем восстановленное изображение
    cv2.imwrite(output_image_path, stitched_image)
    print(f"✅ Восстановленное изображение сохранено: {output_image_path}")
