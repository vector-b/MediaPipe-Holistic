import cv2
import mediapipe as mp
import time
from holistic_utilities import *

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def main():
    #0 for webcam
    #1 for default video
    filename = 'coords.csv'
    videopath = 'testing/tips.mp4'

    #Extrai as coordenadas dos videos indicados
    extract_features(mpPose, mpDraw, mp_holistic, path='./gestures', filename='coords.csv')
    
    #Realiza o treinamento do modelo com o CSV obtido na função anterior
    le, model = training_model_video(filename)

    #Mode - 1 pra video, 2 pra webcam
    mode = 1
    print('Exibindo video teste...')
    generate_video_holistic(le, model,mpPose, mpDraw, mp_holistic, videopath, mode)
    

if __name__ == "__main__":
    main()
