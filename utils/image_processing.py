import os
import cv2
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from metadata import get_image_metadata

PATCH_SIZE = 2048  # Размер патча (2048x2048)
OVERLAP = 512  # Перекрытие патчей (10-20%)


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
            patch_filename = f"{output_dir}/patch_{patch_idx:04d}.png"
            cv2.imwrite(patch_filename, patch)
            patch_paths.append(patch_filename)
            patch_idx += 1

    return patch_paths


def merge_patches(patch_dir: str, output_image_path: str, original_size: Tuple[int, int]):
    """
    Обратно собирает изображение из патчей.

    :param patch_dir: Директория с патчами.
    :param output_image_path: Путь для сохранения восстановленного изображения.
    :param original_size: Оригинальный размер изображения (width, height).
    """
    width, height = original_size
    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)
    weight_map = np.zeros((height, width, 1), dtype=np.float32)  # Для сглаживания границ

    patch_files = sorted(os.listdir(patch_dir))  # Загружаем патчи в правильном порядке
    patch_idx = 0

    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP):
            patch = cv2.imread(os.path.join(patch_dir, patch_files[patch_idx]))
            patch_idx += 1

            # Добавляем патч к изображению
            stitched_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += patch
            weight_map[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1  # Заполняем карту весов

    # Усредняем значения, чтобы избежать резких границ
    weight_map[weight_map == 0] = 1  # Чтобы избежать деления на 0
    stitched_image = (stitched_image / weight_map).astype(np.uint8)

    cv2.imwrite(output_image_path, stitched_image)
