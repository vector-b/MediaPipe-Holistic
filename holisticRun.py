import cv2
import mediapipe as mp
import time
from holistic_utilities import training_model_video, generate_video_holistic

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def main():
    print('Insira o nome/endereço do arquivo csv a ser usado como dataset para nosso modelo:')
    filename = input()
    
    print('Insira o modo de entrada de vídeo - 0 para webcam e 1 para vídeo .mp4')
    mode = input()
    mode = int(mode)

    if(mode):
        print('Insira o endereço do vídeo .mp4 a ser testado:')
        videopath = input()
    else:
        videopath = ''


    #Extrai as coordenadas dos videos indicados
    #Realiza o treinamento do modelo com o CSV obtido na função anterior
    le, model = training_model_video(filename)

    #Mode - 1 pra video, 2 pra webcam
    print('Exibindo video teste...')
    generate_video_holistic(le, model,mpPose, mpDraw, mp_holistic, videopath, mode)

if __name__ == "__main__":
    main()
