import numpy as np

PATCH_SIZE = 200  # Размер патча (2048x2048)
overlap_percent = 0.2
OVERLAP = int(np.ceil(PATCH_SIZE * overlap_percent))  # Перекрытие патчей (10-20%)

