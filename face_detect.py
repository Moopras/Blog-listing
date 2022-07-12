import cv2 as cv

img = cv.imread(r'TrainingImages\\17-52HA066\\Funmi7.jpg')
resized = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=6)
for (x,y,w,h) in faces_rect:
        cv.rectangle(resized, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected face(s)', resized)
print(f'Number of face found= {len(faces_rect)}')
cv.waitKey(0)
















# capture = cv.VideoCapture(0)



# haar_cascade = cv.CascadeClassifier('haar_face.xml')
# while (True):
#     isTrue, img = capture.read()

#     cv.imshow('Detection', img)

#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     cv.imshow('gray', gray)
#     faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=3)

    

#     for (x,y,w,h) in faces_rect:
#         cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

#     cv.imshow('Detected face(s)', img)

    

#     if cv.waitKey(20) & 0xFF==ord('q'):
#              break
           
# print(f'Number of face found= {len(faces_rect)}')

# capture.release()
# cv.destroyAllWindows()
