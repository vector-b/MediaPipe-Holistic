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

    training_with_videos(mpPose, mpDraw, mp_holistic, path='./gestures', filename='coordinates.csv')
    '''path_to_file = 'gestures/5.mp4'
    capture = request_capture(1, path_to_file)
    pTime = 0

    while(1):
        ret , frame = capture.read()

        #mpPose processa cada frame no formato RGB
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        results = holistic.process(frameRGB)

        #R_landmark = results.pose_landmarks

        cTime = time.time()
        
        landmmark_work(results, mpPose, mpDraw, mp_holistic, frame)
        
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (60,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,100,120),3)
        cv2.imshow("Video", frame)


        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()'''


if __name__ == "__main__":
    main()