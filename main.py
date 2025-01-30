import argparse
from utils.image_processing import create_patches, merge_patches
from utils.metadata import get_image_metadata

def main():
    parser = argparse.ArgumentParser(description="Разрезка и сборка изображений.")
    parser.add_argument("--image", type=str, required=True, help="Путь к исходному изображению.")
    parser.add_argument("--output_patches", type=str, default="patches/", help="Папка для сохранения патчей.")
    parser.add_argument("--output_image", type=str, default="reconstructed.png", help="Путь для восстановленного изображения.")
    args = parser.parse_args()

    # 1. Получаем метаданные изображения
    original_size = get_image_metadata(args.image)

    # 2. Разрезаем изображение на патчи
    patch_paths = create_patches(args.image, args.output_patches)
    print(f"Создано {len(patch_paths)} патчей.")

    # 3. Здесь будет процесс обработки патчей (например, загрузка в нейросеть)

    # 4. Обратная сборка изображения
    merge_patches(args.output_patches, args.output_image, original_size)
    print(f"Изображение восстановлено и сохранено в {args.output_image}")

if __name__ == "__main__":
    main()