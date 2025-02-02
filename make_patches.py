import argparse
import os
from utils.image_processing import create_patches, merge_patches
# from utils.metadata import get_image_metadata
from utils.metadata import get_image_metadata
from glob2 import glob
from datetime import datetime

def main(image, output_patches):
    #parser = argparse.ArgumentParser(description="Разрезка и сборка изображений.")
    #parser.add_argument("--image", type=str, required=True, help="Путь к исходному изображению.")
    #parser.add_argument("--output_patches", type=str, default="patches/", help="Папка для сохранения патчей.")
    #parser.add_argument("--output_image", type=str, default="reconstructed.png", help="Путь для восстановленного изображения.")
    #args = parser.parse_args()

    # 1. Получаем метаданные изображения
    original_size = get_image_metadata(image)

    # 2. Разрезаем изображение на патчи
    patch_paths = create_patches(image, output_patches)
    # print(f"Создано {len(patch_paths)} патчей.")

    # 3. Здесь будет процесс обработки патчей (например, загрузка в нейросеть)

    # 4. Обратная сборка изображения
    # merge_patches(output_patches, output_image, original_size)
    # print(f"Изображение восстановлено и сохранено в {output_image}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Разрезка и сборка изображений.")
    # parser.add_argument("--image", type=str, required=True, help="Путь к исходному изображению.")
    # parser.add_argument("--output_patches", type=str, default="patches/", help="Папка для сохранения патчей.")
    # parser.add_argument("--output_image", type=str, default="reconstructed.png", help="Путь для восстановленного изображения.")
    # args = parser.parse_args()

    os.makedirs('data/patches', exist_ok=True)
    os.makedirs('data/patches/raw', exist_ok=True)
    os.makedirs('data/patches/GT', exist_ok=True)
    os.makedirs('data/patches/test', exist_ok=True)
    now = datetime.now()
    output_patches_raw =  './data/patches/raw'       # "Папка для сохранения патчей."
    output_patches_gt =  './data/patches/GT'       # "Папка для сохранения патчей."
    output_patches_test =  './data/patches/test'       # "Папка для сохранения патчей."
    for file in glob('./data/Raw/raw/*'):
        image = file                    # "Путь к исходному изображению."
        main(image, output_patches_raw)
    print(f'Done with train raw patches | time > {datetime.now() - now}')
    now = datetime.now()
    for file in glob('./data/Raw/GT/*'):
        image = file                    # "Путь к исходному изображению."
        main(image, output_patches_gt)
    print(f'Done with raw GT patches | time > {datetime.now() - now}')
    now = datetime.now()
    for file in glob('./data/Raw/test/raw/*'):
        image = file                    # "Путь к исходному изображению."
        main(image, output_patches_test)
    print(f'Done with raw test patches | time > {datetime.now() - now}')