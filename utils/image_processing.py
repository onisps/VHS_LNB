import os
import cv2
import numpy as np
from typing import List, Tuple
from utils.global_variables import PATCH_SIZE, OVERLAP

def create_patches(image_path: str, output_dir: str) -> List[str]:
    """
    Разрезает изображение на патчи с перекрытием и сохраняет их.
    Патчи покрывают всю область исходного изображения, включая его края.
    
    :param image_path: Путь к исходному изображению.
    :param output_dir: Директория для сохранения патчей.
    :return: Список путей к сохранённым патчам.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение по пути {image_path}")
    height, width = image.shape[:2]
    
    # Рассчитываем шаг с учетом перекрытия
    step = PATCH_SIZE - OVERLAP
    
    # Вычисляем координаты верхнего левого угла патчей по вертикали (y)
    y_coords = list(range(0, height - PATCH_SIZE + 1, step))
    if y_coords[-1] != height - PATCH_SIZE:
        y_coords.append(height - PATCH_SIZE)
    
    # Вычисляем координаты верхнего левого угла патчей по горизонтали (x)
    x_coords = list(range(0, width - PATCH_SIZE + 1, step))
    if x_coords[-1] != width - PATCH_SIZE:
        x_coords.append(width - PATCH_SIZE)
    
    patch_paths = []
    patch_idx = 0
    # Проходим по рассчитанным координатам и сохраняем патчи
    for y in y_coords:
        for x in x_coords:
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            patch_filename = os.path.join(output_dir, f"{base_name}-patch_{patch_idx:04d}.png")
            cv2.imwrite(patch_filename, patch)
            patch_paths.append(patch_filename)
            patch_idx += 1
    return patch_paths

def merge_patches(patch_dir: str, output_image_path: str, epoch: int, original_size: Tuple[int, int]):
    """
    Собирает исходное изображение из патчей с учётом перекрытия.
    
    :param patch_dir: Директория с патчами.
    :param output_image_path: Путь для сохранения восстановленного изображения.
    :param epoch: Номер эпохи (используется для формирования имени файла).
    :param original_size: Оригинальный размер изображения в формате (высота, ширина).
    """
    # При загрузке изображения через cv2.imread, shape имеет формат (height, width, channels)
    height, width = original_size
    stitched_image = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # Получаем список файлов патчей, фильтруем по расширениям
    patch_files = sorted([f for f in os.listdir(patch_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    # Группируем патчи по имени исходного изображения (например, "003_0009")
    patch_groups = {}
    for patch_file in patch_files:
        base_name = patch_file.split("-patch_")[0]
        if base_name not in patch_groups:
            patch_groups[base_name] = []
        patch_groups[base_name].append(patch_file)
    
    # Для вычисления координат повторяем ту же схему, что и при разбиении
    step = PATCH_SIZE - OVERLAP
    y_coords = list(range(0, height - PATCH_SIZE + 1, step))
    if y_coords[-1] != height - PATCH_SIZE:
        y_coords.append(height - PATCH_SIZE)
    x_coords = list(range(0, width - PATCH_SIZE + 1, step))
    if x_coords[-1] != width - PATCH_SIZE:
        x_coords.append(width - PATCH_SIZE)
    
    # Обрабатываем каждую группу патчей отдельно
    for base_name, patches in patch_groups.items():
        patch_idx = 0
        # Для каждого рассчитанного положения вставляем соответствующий патч
        for y in y_coords:
            for x in x_coords:
                patch_path = os.path.join(patch_dir, patches[patch_idx])
                patch = cv2.imread(patch_path)
                if patch is None:
                    raise ValueError(f"Не удалось загрузить патч: {patch_path}")
                patch = patch.astype(np.float32)
                patch_idx += 1
                
                # Создаём маску весов для сглаживания перекрытий
                patch_weight = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                if OVERLAP > 0:
                    # Горизонтальные края
                    patch_weight[:OVERLAP, :] *= np.linspace(0, 1, OVERLAP)[:, np.newaxis]
                    patch_weight[-OVERLAP:, :] *= np.linspace(1, 0, OVERLAP)[:, np.newaxis]
                    # Вертикальные края
                    patch_weight[:, :OVERLAP] *= np.linspace(0, 1, OVERLAP)
                    patch_weight[:, -OVERLAP:] *= np.linspace(1, 0, OVERLAP)
                
                stitched_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += patch * patch_weight[:, :, np.newaxis]
                weight_map[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += patch_weight
        
        # Избегаем деления на ноль
        weight_map[weight_map == 0] = 1  
        merged = (stitched_image / weight_map[:, :, np.newaxis]).astype(np.uint8)
        
        # Сохраняем восстановленное изображение
        os.makedirs(output_image_path, exist_ok=True)
        output_file = os.path.join(output_image_path, f'{base_name}_{epoch}.jpg')
        cv2.imwrite(output_file, merged)
        # print(f"✅ Восстановленное изображение сохранено: {output_file}")
