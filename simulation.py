import cv2
import numpy as np

def simulate_cataract(image):
    """Simula a visão com catarata (visão embaçada)."""
    # Aplica um desfoque gaussiano intenso para simular o embaçamento
    return cv2.GaussianBlur(image, (31, 31), 0)

def simulate_glaucoma(image):
    """Simula a visão com glaucoma (perda de visão periférica)."""
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Cria um círculo no centro que representa a área de visão restante
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y) // 2
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Aplica a máscara à imagem
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def simulate_macular_degeneration(image):
    """Simula a degeneração macular (mancha escura no centro da visão)."""
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    
    # Desenha um círculo preto no centro para simular a perda de visão central
    radius = min(center_x, center_y) // 4
    simulated_image = image.copy()
    cv2.circle(simulated_image, (center_x, center_y), radius, (0, 0, 0), -1)
    
    return simulated_image

def get_simulation_for_disease(disease_name):
    """Retorna a função de simulação correspondente e uma descrição."""
    simulations = {
        'cataract': {
            'function': simulate_cataract,
            'description': "A catarata causa um embaçamento da lente do olho, resultando em uma visão turva e 'leitosa'."
        },
        'glaucoma': {
            'function': simulate_glaucoma,
            'description': "O glaucoma danifica o nervo óptico, geralmente causando uma perda progressiva da visão periférica (visão de túnel)."
        },
        'macular_degeneration': {
            'function': simulate_macular_degeneration,
            'description': "A degeneração macular afeta a mácula, a parte central da retina, causando uma mancha escura ou distorção no centro da visão."
        },
        'normal': {
            'function': lambda img: img,  # Nenhuma simulação para visão normal
            'description': "Visão normal e saudável, sem obstruções ou distorções."
        }
    }
    return simulations.get(disease_name, simulations['normal'])
