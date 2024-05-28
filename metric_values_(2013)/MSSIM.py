import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
def calculate_mssim(image1_path, image2_path):
    # Carregar as imagens
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Calcular o MSSIM
    mssim_index = ssim(img1, img2, full=True)

    return mssim_index

# Substitua 'caminho_da_imagem1.jpg' e 'caminho_da_imagem2.jpg' pelos caminhos reais das suas imagens
