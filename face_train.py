import os
import cv2 as cv
import numpy as np


def create_train():
    p=[]
    for i in os.listdir(r'C:\OpenCV2\TrainingImages'):
        p.append(i)

    print(p)
    DIR = r'C:\OpenCV2\TrainingImages'
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    features = []
    labels = []
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            resized = cv.resize(img_array, (800,500), interpolation = cv.INTER_AREA)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
    print('Training done ---------------------')

    features = np.array(features, dtype = 'object')
    labels = np.array(labels)

    print(f'Length of features = {len(features)}') 
    print(f'Length of labels = {len(labels)}')

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)
    face_recognizer.save('face_trained.yml')
    np.save('features.npy', features)
    np.save('labels.npy', labels)

create_train()

