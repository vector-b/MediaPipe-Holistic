import cv2
import mediapipe as mp
import time
import os
import csv
import numpy as np

num_coords = 501

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


def training_with_videos(mpPose, mpDraw, mpHol, path=None, filename=None):

    holistic = mpHol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #path == root of directory with all classes
    if (path is None):
        path = './gestures'
    if (filename is None):
        filename = 'coord.csv'
    
    dir = path
    sub_dir = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

    landmarks_class = ['class']
    for i in range (1, num_coords+1):
        landmarks_class += ['x{}'.format(i), 'y{}'.format(i) , 'z{}'.format(i), 'v{}'.format(i)]

    with open(filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks_class)
    
    for i in sub_dir:
        class_name = i
        path_to_file = dir+'/'+ i
        files = [name for name in os.listdir(path_to_file) if os.path.isfile((os.path.join(path_to_file, name)))]
        for j in files:
            filepath = path_to_file+'/'+ j
            capture = request_capture(1, filepath)
            pTime = 0
            while(1):
                ret , frame = capture.read()

                #mpPose processa cada frame no formato RGB

                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False


                    results = holistic.process(image)


                    image.flags.writeable = True   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    #R_landmark = results.pose_landmarks

                    cTime = time.time()
                
                    landmmark_work(results, mpPose, mpDraw, mpHol, frame)
                
                    fps = 1/(cTime - pTime)
                    pTime = cTime
                
                    cv2.putText(frame, str(int(fps)), (60,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,100,120),3)
                    cv2.imshow("Video", frame)


                    try:
                        # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                        if(results.face_landmarks):
                            face = results.face_landmarks.landmark
                            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                        else:
                            face_row = list(np.zeros(468*4).flatten())
                    
                        # Concate rows
                        row = pose_row+face_row
                    
                        # Append class name 
                        row.insert(0, class_name)
                    
                        # Export to CSV
                        with open(filename, mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row) 

                    except:
                        pass
                except:
                   break


                k = cv2.waitKey(5) & 0xff
                if k == 27:
                    break

            capture.release()
            cv2.destroyAllWindows()

def predict_video_hls(mpPose, mpDraw, mpHol, path):
           
        
        


