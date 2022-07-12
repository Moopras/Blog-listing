import tkinter as tk
from tkinter import Message, Text
from unittest.mock import CallableMixin
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
from threading import *
import os
from mainrt import takeAtten

window = tk.Tk()
window.title("Face_Recogniser")
window.configure(background = 'white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Face Identification-Based Attendance Monitoring System",
    bg ="#00008B", fg = "#FFFF00", width = 60,
    height = 3, font = ('times', 20, 'bold'))

message.place(x = 200, y = 20)

lbl = tk.Label(window, text = "Matric No.",
width = 20, height = 2, fg ="#00008B",
bg = "white", font = ('times', 15, ' bold ') )
lbl.place(x = 400, y = 200)

txt = tk.Entry(window,
width = 20, bg ="white",
fg ="#00008B", font = ('times', 15, ' bold '))
txt.place(x = 700, y = 215)

lbl2 = tk.Label(window, text ="Name",
width = 20, fg ="#00008B", bg ="white",
height = 2, font =('times', 15, ' bold '))
lbl2.place(x = 400, y = 300)

txt2 = tk.Entry(window, width = 20,
bg ="white", fg ="#00008B", 
font = ('times', 15, ' bold ') )
txt2.place(x = 700, y = 315)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():	

    Id = txt.get()
    name =txt2.get()

    if not Id.isnumeric() or  not name.isalpha():
        res = "Please check details"
        message.configure(text = res)
        return
    else:
        video = r'C:\OpenCV2\TrainingImages\Video'
        video_images = 'C:\\OpenCV2\\TrainingImages\\'+ Id +'\\' + name

        if not os.path.isdir(video_images):
            os.makedirs(video_images)

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            exit(0)
        harcascadePath = r"C:\OpenCV2\haar_face.xml"
        haar_cascade = cv2.CascadeClassifier(harcascadePath)
        frameFrequency=25
        total_frame = 0
        id = 1
        while True:
            if id == 21 or id > 20:
                break
            ret, frame = cap.read()
            if ret is False:
                break
            total_frame += 1
            if total_frame % frameFrequency == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                faces = haar_cascade.detectMultiScale(gray, 1.1, 6)
                        
                    
            image_name = video_images + str(id) +'.jpg'

            cv2.imwrite(image_name, frame)
            print(image_name)
            id +=1
                
        cap.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id +" Name : "+ name
        message.configure(text = res)

def threadingTkImg():
    TakeImgthrd = Thread(target =TakeImages)
    TakeImgthrd.start()

def TrainImages():
    p=[]
    for i in os.listdir(r'C:\OpenCV2\TrainingImages'):
        p.append(i)

    print(p)
    DIR = r'C:\OpenCV2\TrainingImages'
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    features = []
    labels = []
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            resized = cv2.resize(img_array, (800,500), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
    print('Training done ---------------------')

    features = np.array(features, dtype = 'object')
    labels = np.array(labels)


    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)
    face_recognizer.save('face_trained.yml')
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    res = "Images Trained"
    message.configure(text = res)
            

def attendance(name):
    with open (r'C:\OpenCV2\Attendance\Attendance.csv', 'r+') as f:
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


def TakeAttendance():
    takeAtten()
    # p=[]
    # for i in os.listdir(r'C:\Users\Psalm johnson\Documents\Dev\OpenCV2\TrainingImages'):
    #     print(p)
    # harcascadePath = r"C:\Users\Psalm johnson\Documents\Dev\OpenCV2\haar_face.xml"
    # capture = cv2.VideoCapture(0)
    # while True:
    #     haar_cascade = cv2.CascadeClassifier(harcascadePath)
            
    #     face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #     face_recognizer.read(r'C:\Users\Psalm johnson\Documents\Dev\OpenCV2\TrainingImageLabel\face_trained.yml')
    #     if not capture.isOpened():
    #         print("not opened..")
    #         exit(0)
    #     isTrue, frame = capture.read()
    #     cv2.imshow('Webcam', frame)
        
    #     resized = cv2.resize(frame, (800, 600), interpolation = cv2.INTER_AREA)
    #     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #     faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
    #     for (x,y,w,h) in faces_rect:
    #         faces_roi = gray[y:y+h, x:x+h]
    #         label, confidence = face_recognizer.predict(faces_roi)
    #         if confidence < 50 :
    #             cv2.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), thickness=1)
    #             cv2.putText(resized, "Unknown", (x,y-10), 
    #             cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), thickness = 1)
    #         elif confidence > 100:
    #             cv2.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), thickness=1)
    #             cv2.putText(resized, "Unknown", (x,y-10), 
    #             cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), thickness = 1)
    #         else:
    #             cv2.rectangle(resized, (x,y), (x+w, y+h), (0,255,0), thickness = 2)
    #             cv2.putText(resized, str(p[label]), (x, y-20), 
    #             cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), thickness = 2)
    #             print(f'Label = {p[label]} with a confidence of {confidence}')

    #         cv2.imshow('Recognized Student', resized)
            
    #     if cv2.waitKey(20) & 0xFF==ord('q'):
    #         break
        
    #     capture.release()
    #     cv2.destroyAllWindows()
    #attendance(p[label])
        

takeImg = tk.Button(window, text ="Take New Images",
command = threadingTkImg, fg ="#FFFF00", bg ="#00008B",
width = 20, height = 3, activebackground = "#00008B", activeforeground= "white",
font =('times', 15, ' bold '))
takeImg.place(x = 100, y = 500)
trainImg = tk.Button(window, text ="Training",
command = TrainImages, fg ="#FFFF00", bg ="#00008B",
width = 20, height = 3, activebackground = "#00008B", activeforeground= "white",
font =('times', 15, ' bold '))
trainImg.place(x = 400, y = 500)
trackImg = tk.Button(window, text ="Take Attendance",
command = TakeAttendance, fg ="#FFFF00", bg ="#00008B",
width = 20, height = 3, activebackground = "#00008B", activeforeground= "white",
font =('times', 15, ' bold '))
trackImg.place(x = 700, y = 500)
quitWindow = tk.Button(window, text ="Quit",
command = window.destroy, fg ="#FFFF00", bg ="#00008B",
width = 20, height = 3, activebackground = "#00008B", activeforeground= "white",
font =('times', 15, ' bold '))
quitWindow.place(x = 1000, y = 500)


window.mainloop()
