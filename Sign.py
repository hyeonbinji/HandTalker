import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

max_num_hands = 1 #최대 손 인식 개수

gesture = {
    0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',
    8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',
    15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',
    22:'w',23:'x',24:'y',25:'z'
}


mp_hands = mp.solutions.hands #연두색으로 손가락 마디 표시
mp_drawing = mp.solutions.drawing_utils #점으로 손가락 마디 표시
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5, #
    min_tracking_confidence = 0.5 #
) #mp_hands.Hands: 손가락 인식하는 모듈 초기화

f = open('test.txt','w')

file = np.genfromtxt('hand_data_with_labels.txt',delimiter = ',') #csv파일 불러오는 함수: genfromtxt
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

knn = cv2.ml.KNearest_create() #OpenCV에서 KNN 학습시키기, KNN: 분류학습기
knn.train(angle,cv2.ml.ROW_SAMPLE,label) #cv2.ml.ROW_SAMPLE: 학습 데이터가 행 단위로 구성됨
cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1
while True:
    ret,img = cap.read() #cap.read로 카메라로부터 프레임 읽어오기, ret: 잘 읽었는지 유뮤 T/F, img: 읽어온 이미지
    if not ret:
        continue
    img = cv2.flip(img,1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result = hands.process(img) #얘로 손 인식한다

    if result.multi_hand_landmarks is not None: #손을 인식했으면
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3)) #빨간 점들이 joint(21개), 그리고 joint별 x,y,z좌표를 저장한다 .
            for j,lm in enumerate(res.landmark):
                joint[j] = [lm.x,lm.y,lm.z]
            #https: // developers.google.com / mediapipe / solutions / vision / hand_landmarker
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1
            #Nomalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            #Dot product의 아크코사인으로 각도를 구한다.
            compareV1 = v[[0,1,2,4,5,6,7,8,9,10,12,13,14,16,17],:]
            compareV2 = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]
            angle = np.arccos(np.einsum('nt,nt->n',compareV1,compareV2))
            angle = np.degrees(angle) #radian값을 degree로 변환

            data = np.array([angle],dtype = np.float32)
            ret,results,neighbours,dist = knn.findNearest(data,3) #K = 3일때 결과 구하기
            idx = int(results[0][0])

            if idx in gesture.keys():
                if idx != prev_index:
                    startTime = time.time()
                    prev_index = idx
                else:
                    if time.time() - startTime > recognizeDelay: #recognizeDelay: 이 시간보다 길면 인식한다.
                        if idx == 26:
                            sentence += ' '
                        elif idx == 27:
                            sentence = ''
                        else:
                            sentence += gesture[idx]
                        startTime = time.time()

                #인식된 손동작을 화면에 표시한다.
                cv2.putText(img, gesture[idx].upper(), (int(res.landmark[0].x * img.shape[1] - 10),
                                                        int(res.landmark[0].y * img.shape[0] + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
        cv2.putText(img,sentence,(20,440),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3) #좌측 상단에 표시한다.

        cv2.imshow('HandTracking',img)
        if cv2.waitKey(1) == ord('q'):
            break

