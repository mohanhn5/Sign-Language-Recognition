import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import pyttsx3
engine = pyttsx3.init()

model = load_model('smnist.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

img_counter = 0
tempWord=''
word=''
analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
while True:
    _, frame = cap.read()
    k = cv2.waitKey(1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            pre_text=''
                
            analysisframe = frame
            showframe = analysisframe
            cv2.imshow("Frame", showframe)
            framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
            resultanalysis = hands.process(framergbanalysis)
            hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
            if hand_landmarksanalysis:
                for handLMsanalysis in hand_landmarksanalysis:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lmanalysis in handLMsanalysis.landmark:
                        x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    y_min -= 20
                    y_max += 20
                    x_min -= 20
                    x_max += 20  
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe,(28,28))


            nlist = []
            rows,cols = analysisframe.shape
            for i in range(rows):
                for j in range(cols):
                    k = analysisframe[i,j]
                    nlist.append(k)
            
            datan = pd.DataFrame(nlist).T
            colname = []
            for val in range(784):
                colname.append(val)
            datan.columns = colname

            pixeldata = datan.values
            pixeldata = pixeldata / 255
            pixeldata = pixeldata.reshape(-1,28,28,1)
            prediction = model.predict(pixeldata)

            predarray = np.array(prediction[0])
            letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]

            key_list=list(letter_prediction_dict.keys())
            val_list=list(letter_prediction_dict.values())
            ind=val_list.index(high1)
            pre_text += key_list[ind]
            print(pre_text)
            cv2.putText(frame, pre_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                cv2.LINE_AA)
                   
            tempWord=tempWord+pre_text
            
    cv2.putText(frame, word, (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                cv2.LINE_AA)
    if k%256 ==32: #entering space to save a character
        most_frequent = max(set(tempWord), key = tempWord.count)
        tempWord=''
        word=word+most_frequent
    elif k%256 ==115: #entering 's' to translate into speech
        engine.setProperty('rate', 125)
        engine.say(word)
        engine.runAndWait()
    elif k%256 ==99: #entering 'c' to clear word
        word=''
    elif k%256 ==122: #entering 'z' to undo
        word = word[:-1]
    elif k%256 == 27: #entering ESC to close window
        break

    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()