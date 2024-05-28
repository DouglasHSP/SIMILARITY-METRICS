import cv2
import numpy as np

def calculate_nqm(original_image_path, compressed_image_path):
    """
    Calcula a métrica de qualidade de imagem Normalized Quantization Metric (NQM) a partir de arquivos de imagem.

    Args:
    original_image_path: O caminho para a imagem original.
    compressed_image_path: O caminho para a imagem comprimida.

    Returns:
    nqm_score: O valor NQM calculado.
    """
    # Função interna para calcular NQM
    def calculate_nqm_internal(original_image, compressed_image):
        # Verificar se as imagens têm as mesmas dimensões
        if original_image.shape != compressed_image.shape:
            raise ValueError("As dimensões das imagens original e comprimida devem ser iguais.")

        # Normalizar as imagens para valores entre 0 e 1
        original_image = original_image.astype(float) / 255.0
        compressed_image = compressed_image.astype(float) / 255.0

        # Calcular a diferença absoluta entre os pixels
        diff = original_image - compressed_image

        # Calcular o NQM
        mse = np.mean(diff ** 2)
        max_val = np.max(original_image)
        nqm_score = mse / (max_val ** 2)

        return nqm_score
    
    # Carregar as imagens usando OpenCV
    original_image = cv2.imread(original_image_path)
    compressed_image = cv2.imread(compressed_image_path)

    # Verificar se as imagens foram carregadas corretamente
    if original_image is None or compressed_image is None:
        raise FileNotFoundError("Não foi possível carregar uma ou ambas as imagens.")

    # Calcular e retornar o NQM
    return calculate_nqm_internal(original_image, compressed_image)

# Exemplo de uso:
# Suponha que 'original_image_path' e 'compressed_image_path' sejam os caminhos para os arquivos de imagem.
# nqm = calculate_nqm(original_image_path, compressed_image_path)
