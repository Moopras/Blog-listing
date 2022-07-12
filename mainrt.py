import numpy as np
import os
import cv2 as cv
from datetime import datetime


def takeAtten():
     capture = cv.VideoCapture(0)
     while True:
          isTrue, frame = capture.read()
          cv.imshow('Webcam', frame)
          haar_cascade = cv.CascadeClassifier(r'C:\Users\Psalm johnson\Documents\Dev\OpenCV2\haar_face.xml')
          people = ['17--52HA096', '17-52HA074','17-52HA066']
     # features = np.load('features.npy')
     # labels = np.load('labels.npy')
          face_recognizer = cv.face.LBPHFaceRecognizer_create()
          face_recognizer.read(r'C:\Users\Psalm johnson\Documents\Dev\OpenCV2\\face_trained.yml')
          resized = cv.resize(frame, (800, 600), interpolation = cv.INTER_AREA)
          gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
          faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
          for (x,y,w,h) in faces_rect:
               faces_roi = gray[y:y+h, x:x+h]
               label, confidence = face_recognizer.predict(faces_roi)
               if confidence < 50 :
                    cv.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), thickness=1)
                    cv.putText(resized, "Unknown", (x,y-10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), thickness = 1)
               elif confidence > 100:
                    cv.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), thickness=1)
                    cv.putText(resized, "Unknown", (x,y-10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), thickness = 1)
               else:
                    cv.rectangle(resized, (x,y), (x+w, y+h), (0,255,0), thickness = 2)
                    cv.putText(resized, str(people[label]), (x, y-20), cv.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), thickness = 2)
                    print(f'Label = {people[label]} with a confidence of {confidence}')

               cv.imshow('Recognized Student', resized)

          if cv.waitKey(20) & 0xFF==ord('q'):
               break
     capture.release()
     cv.destroyAllWindows()
     def attendance(name):
          with open (r'C:\Users\Psalm johnson\Documents\Dev\OpenCV2\Attendance\Attendance.csv', 'r+') as f:
               myDataList = f.readlines()
               nameList = []
               for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
               if name not in nameList:
                    now = datetime.now()
                    timestr = now.strftime('%H:%M:%S')
                    date = now.strftime(f'%a %B %dth %Y')
                    f.writelines(f'\n{name}, {timestr}, {date}')

     attendance(people[label])

takeAtten()





