from fileinput import filename
import cv2
import mediapipe as mp
import time
from holistic_utilities import *

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    


def main():
    #0 for webcam
    #1 for default video
    filename = 'coord.csv'
    videopath = 'gestures/testing/test.mp4'

    #Extrai as coordenadas dos videos indicados
    extract_features(mpPose, mpDraw, mp_holistic, path='./gestures', filename='coordinates.csv')
    
    #Realiza o treinamento do modelo com o CSV obtido na função anterior
    le, model = training_model_video(filename)

    #Mode - 1 pra video, 2 pra webcam
    mode = 1
    print('Exibindo video teste...')
    generate_video_holistic(le, model,mpPose, mpDraw, mp_holistic, videopath, mode)
    

if __name__ == "__main__":
    main()