import cv2
import numpy as np
def calcular_vsnr(img1, img2):
    # Carregando as imagens
    img1 = cv2.imread(img1).astype(np.float64)
    img2 = cv2.imread(img2).astype(np.float64)
    
    # Calculando a diferença entre as imagens
    diff = img1 - img2
    
    # Calculando o erro quadrático médio (MSE) entre as imagens para cada canal de cor
    mse_r = np.mean(diff[:,:,0] ** 2)
    mse_g = np.mean(diff[:,:,1] ** 2)
    mse_b = np.mean(diff[:,:,2] ** 2)
    
    # Média do MSE dos três canais de cor
    mse = (mse_r + mse_g + mse_b) / 3
    
    # Definindo a constante de sensibilidade visual
    k = 255
    
    # Calculando o VSNR
    vsnr_value = 20 * np.log10(k / np.sqrt(mse))
    
    return vsnr_value