# VHS / Virtual Hystology Staining /
## CUT: Виртуальное окрашивание гистологических изображений

Этот проект реализует **виртуальное окрашивание гистологических изображений** с помощью **Generative adversarial networks (GAN)**.  
Используется **модульный подход**, обеспечивающий:
- Разрезку больших изображений на патчи (`PATCH_SIZE` с перекрытием `overlap_percent`  в `utils/global_variables.py`)
- Обучение сетки на патчах  
- Сохранение, загрузку с чекпойнта и применение модели  
- **Callbacks** для логирования и сохранения чекпоинтов  

---

## 🚀 **1. Установка**
### 📌 **1.1. Установите зависимости**
```sh
pip install torch torchvision torchmetrics tensorboard tqdm numpy opencv-python pillow
```

### 📌 **1.2. Клонируй репозиторий**
```sh
git clone https://github.com/onisps/VHS_LNB.git
cd VHS_LNB
```

## 🎓 **2. Обучение модели**
### 🎓 **2.1. Запуск обучения**
Перед обучением необходимо разрезать большие изображения на патчи.
```sh
nohup python train_gan.py &
```

Что делает этот шаг?
- **Разрезает изображение на патчи `PATCH_SIZE x PATCH_SIZE` с `overlap_percent` перекрытием**
- **Сохраняет их в `data/patches/`**
- **Обучает модель**
- **Сохраняет чекпоинты модели в checkpoints/**
- **Сохраняет чекпойнты, применяет модель на текущем шаге к `data/patches/test/` и собрает обратно в `data/test_model/`**
- **Логирует метрики в TensorBoard**

### 🎓 **2.2. Просмотр логов TensorBoard**
```sh
tensorboard --logdir=logs --host 0.0.0.0
```

Перейдите в браузер: http://localhost:6006/

## 💾 **3. Сохранение и загрузка модели**
### 💾 **3.1. Сохранение модели после обучения**
Модель автоматически сохраняется, но можно сделать это вручную:

```python
from utils.model_utils import save_model

save_model(model, optimizer, epoch)
```

### 💾 **3.2. Загрузка модели**

```python
from utils.model_utils import load_model

model = load_model(f"checkpoints/{model_name}_{epoch}.pth")
```

### 🎨 **4. Применение модели к новому изображению**
```sh
python apply_model.py
```

Что делает этот шаг?
- **Загружает обученную модель**
- **Преобразует изображение в Tensor**
- **Пропускает его через модель**
- **Сохраняет финальное изображение**

### 📂 **6. Структура проекта**
```
📦
├── 📂 checkpoints/              # Чекпоинты обученной модели
├── 📂 logs/                     # Логи обучения для TensorBoard
├── 📂 data/                     # Данные для обучения
|   ├── 📂 Raw/                  # Сырые неразрезанные данные
|       ├── 📂 GT/
|       ├── 📂 raw/
|       ├── 📂 test/
|   ├── 📂 patches/              # Патчи
|       ├── 📂 raw/
|       ├── 📂 GT/
|       ├── 📂 test/
|   ├── 📂 test_model/           # Сырые неразрезанные данные
├── 📂 utils/                    # Вспомогательные модули
│   ├── 🏗 cut_model.py          # Архитектура модели CUT
│   ├── 🏗 cycle_gan_model.py    # Архитектура модели C-GAN
│   ├── 🏗 global_variables.py   # глобальные переменные проекта
│   ├── 📦 dataset.py            # Датасет для обучения
│   ├── 🖼 image_processing.py   # Разрезка и сборка изображений
│   ├── 📝 metadata.py           # Получение информации об изображениях
│   ├── 💾 model_utils.py        # Функции сохранения/загрузки модели
├── 🖼 apply_model.py            # Применение обученной модели
├── 🖼 make_patches.py           # Разрезка изображений на патчи
├── 🎓 train.py                  # Обучение модели CUT
├── 🎓 train_gan.py              # Обучение модели GAN
├── 📜 README.md                 # Документация проекта

```