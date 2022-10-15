import cv2
import mediapipe as mp
import time
import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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


def extract_features(mpPose, mpDraw, mpHol, path=None, filename=None):

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

def training_model_video(filename,):
    print('Lendo arquivo CSV...')
    data = pd.read_csv(filename)

    le = preprocessing.LabelEncoder()

    data['class'] = le.fit_transform(data['class'])

    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]

    print('Separando conjuntos de Treino e teste..')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

    from xgboost import XGBClassifier

    xgb = XGBClassifier(n_estimators = 1000)
    
    print('Treinando o Classificador...')
    xgb.fit(X_train, y_train)

    predicted = xgb.predict(X_test)
    f1 = f1_score(predicted,y_test, average='macro')
    print('F1-Score: {}'.format(f1))

    return le, xgb

def generate_video_holistic(le, model,mpPose, mpDraw, mpHol, videopath, mode):
    holistic = mpHol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    path_to_file = videopath
    capture = request_capture(mode, path_to_file)
    pTime = 0
    while(1):
        ret , frame = capture.read()

        try:
            #mpPose processa cada frame no formato RGB
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            results = holistic.process(frameRGB)

            #R_landmark = results.pose_landmarks

            cTime = time.time()
            
            landmmark_work(results, mpPose, mpDraw, mpHol, frame)
            
            fps = 1/(cTime - pTime)
            pTime = cTime

            cv2.putText(frame, str(int(fps)), (60,60), cv2.FONT_HERSHEY_PLAIN, 3, (250,0,90),3)
            

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                if(results.face_landmarks):
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                else:
                    face_row = list(np.zeros(468*4).flatten())
                # Concate rows
                
                row = pose_row+face_row
                print('passou da row')

                
                X = np.array(row).reshape(1,-1)
                body_language_class = le.inverse_transform(model.predict(X))
                body_language_prob = model.predict_proba(X)

                # Caixa de Info
                #cv2.rectangle(frame, (50,380), (250, 130), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(frame, 'CLASS'
                            , (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, body_language_class[0]
                            , (50,430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 233), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(frame, 'PROBABILITY'
                            , (50,480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(round(body_language_prob.max(),2))
                            , (50,510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 233), 2, cv2.LINE_AA)
            except:
                pass

        except:
            break

        cv2.imshow("Video", frame)

        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

    
        


