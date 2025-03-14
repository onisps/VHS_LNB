import numpy as np

PATCH_SIZE =  128
overlap_percent = 0.2
OVERLAP = int(np.ceil(PATCH_SIZE * overlap_percent))  # Перекрытие патчей (10-20%)
LR = 1e-4
