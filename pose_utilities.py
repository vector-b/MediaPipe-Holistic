import cv2
import mediapipe as mp
import time



def landmmark_work(results, mpPose, mpDraw,mpHol, frame):
    if (results.pose_landmarks):
        mpDraw.draw_landmarks(frame, results.face_landmarks, mpHol.FACEMESH_TESSELATION, 
                                 mpDraw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mpDraw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mpDraw.draw_landmarks(frame, results.right_hand_landmarks, mpHol.HAND_CONNECTIONS, 
                                 mpDraw.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mpDraw.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mpDraw.draw_landmarks(frame, results.left_hand_landmarks, mpHol.HAND_CONNECTIONS, 
                                 mpDraw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mpDraw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpHol.POSE_CONNECTIONS, 
                                 mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

def request_capture(type=0, path=None):

    if(type):
        capture = cv2.VideoCapture(path)
    else:
        capture = cv2.VideoCapture(0)
    return capture


''' mpDraw.draw_landmarks(frame, R_landmark, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(R_landmark.landmark):
            h, w, c = frame.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (20,120,20), -1 )'''