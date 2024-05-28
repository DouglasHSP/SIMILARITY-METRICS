import numpy as np
import cv2
def calc_psnr_ha(img1, img2, max_val=255):
    # Convertendo as imagens para arrays numpy, se necessário
    imagem1= cv2.imread(img1)
    imagem2= cv2.imread(img2)
    imagem1 = np.array(imagem1)
    imagem2 = np.array(imagem2)
    
    # Calculando o erro quadrático médio (MSE)
    mse = np.mean((imagem1 - imagem2) ** 2)
    
    # Calculando o PSNR
    psnr = 10 * np.log10((max_val ** 2) / mse)
    
    # Calculando o histograma
    hist1, _ = np.histogram(imagem1, bins=256, range=(0, max_val))
    hist2, _ = np.histogram(imagem2, bins=256, range=(0, max_val))
    
    # Calculando a similaridade do histograma
    hist_sim = np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)
    
    # Calculando o PSNR-HA
    psnr_ha = psnr * hist_sim
    
    return psnr_ha