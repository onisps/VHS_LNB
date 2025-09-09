import numpy as np

PATCH_SIZE =  256
overlap_percent = 0.2
OVERLAP = 64# int(np.ceil(PATCH_SIZE * overlap_percent))  # Перекрытие патчей (10-20%)
LR = 1e-4
LEARNING_RATE_G = 0.0002  # Suggestion: Adjust learning rates
LEARNING_RATE_D = 0.0002
BETA1 = 0.5  # Suggestion: Adam optimizer beta1
BETA2 = 0.999 # Suggestion: Adam optimizer beta2
LAMBDA_CYCLE = 10.0 # Suggestion: Emphasize cycle consistency
LAMBDA_IDENTITY = 0.5 # Suggestion: Identity loss weight
