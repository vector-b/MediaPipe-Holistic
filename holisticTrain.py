import cv2
import mediapipe as mp
import time
from holistic_utilities import extract_features

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def main():
    print('Insira o endereço da pasta onde estão todas as classes de imagens/vídeos:')
    path = input()

    print('Insira o nome do arquivo output das features (csv):')
    filename = input()

    #Extrai as coordenadas dos videos indicados
    print('Iniciando extração...')
    extract_features(mpPose, mpDraw, mp_holistic, path, filename=filename)

if __name__ == "__main__":
    main()
