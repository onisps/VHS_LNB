import numpy as np

PATCH_SIZE =  64
overlap_percent = 0.2
OVERLAP = int(np.ceil(PATCH_SIZE * overlap_percent))  # Перекрытие патчей (10-20%)

