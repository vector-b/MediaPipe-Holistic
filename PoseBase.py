from traceback import print_tb
from unittest import result
import cv2
import mediapipe as mp
import time

from regex import R


mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()







#Instancia a captura do arquivo desejado
capture = cv2.VideoCapture('gestures/5.mp4')

#Tempo Inicial, utilizado para calculo de fps
pTime = 0
while(1):
    ret , frame = capture.read()

    #mpPose processa cada frame no formato RGB
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    results = pose.process(frameRGB)

    #Imprime os pontos focais da posição
    #print(results.pose_landmarks)

    R_landmmark = results.pose_landmarks
    #Caso nesse frame tenha landmarks, imprime na tela os pontos e conexões 
    if (R_landmmark):
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(R_landmmark.landmark):
            h, w, c = frame.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (20,120,20), -1 )

    cTime = time.time()
    

    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (60,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,100,120),3)


    cv2.imshow("Video", frame)

    #Press ESC to leave
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()

