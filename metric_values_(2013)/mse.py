import cv2
import numpy as np

def calculate_mse(img1, img2):
    # Carregar as imagens
    img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    
    # Verificar se as imagens foram carregadas corretamente
    if img1 is None or img2 is None:
        raise ValueError("Erro ao carregar imagens")
    
    # Verificar se as imagens têm as mesmas dimensões
    if img1.shape != img2.shape:
        raise ValueError("As imagens devem ter as mesmas dimensões")
    
    # Calcular o MSE entre as imagens
    mse = np.mean((img1 - img2)**2)
    return mse