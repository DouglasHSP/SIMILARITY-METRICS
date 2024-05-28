from skimage.metrics import structural_similarity as ssim
import cv2
def calcular_ssim(caminho_imagem1, caminho_imagem2):
    # Carregar as imagens
    imagem1 = cv2.imread(caminho_imagem1)
    imagem2 = cv2.imread(caminho_imagem2)

    # Verificar se as imagens foram carregadas corretamente
    if imagem1 is None or imagem2 is None:
        raise ValueError("Não foi possível carregar uma ou ambas as imagens.")

    # Converter as imagens para escala de cinza
    imagem1_gray = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
    imagem2_gray = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)

    # Calcular o SSIM
    valor_ssim, _ = ssim(imagem1_gray, imagem2_gray, full=True)

    return valor_ssim