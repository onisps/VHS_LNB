from PIL import Image
from typing import Tuple


def get_image_metadata(image_path: str) -> Tuple[int, int]:
    """
    Получает размер исходного изображения из его метаданных.

    :param image_path: Путь к изображению
    :return: (ширина, высота) изображения
    """
    with Image.open(image_path) as img:
        return img.size  # (width, height)
