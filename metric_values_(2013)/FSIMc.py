import cv2
import numpy as np

def calc_fsimc(image1_path, image2_path):
    # Carregar as imagens
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Verificar se as imagens foram carregadas corretamente
    if img1 is None or img2 is None:
        print("Erro ao carregar as imagens.")
        return None

    # Convertendo as imagens para escala de cinza
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Calculando o gradiente das imagens
    grad_x1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0)
    grad_y1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1)
    grad_mag1 = cv2.magnitude(grad_x1, grad_y1)

    grad_x2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0)
    grad_y2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1)
    grad_mag2 = cv2.magnitude(grad_x2, grad_y2)

    # Calculando a estrutura de borda
    structure_similarity = (2 * grad_mag1 * grad_mag2 + 1e-8) / (grad_mag1 ** 2 + grad_mag2 ** 2 + 1e-8)

    # Calculando o mapa de erro
    err_map = (gray1 - gray2) ** 2

    # Calculando a medida de similaridade de textura
    texture_similarity = np.exp(-np.mean(err_map) / (2 * np.std(err_map) ** 2 + 1e-8))

    # Calculando a medida de similaridade global
    alpha = 0.5  # Fator de peso para a estrutura de borda
    similarity = alpha * structure_similarity + (1 - alpha) * texture_similarity

    return similarity