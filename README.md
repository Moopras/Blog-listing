# Face-Recognition-System-For-Class-Attendance-Monitoring
This repository contains the project I submitted for my final year in school. 

I used Open Source Computer Vision (OpenCV), Python and tkinter gui interface. 
The software has the ability to take images, train them and use the trained images to take attendance. 

For a new user, the user enters the name and matric number of the student, then clicks the face detection button.
When the Face detect button is clicked, the system opens the chosen webcam and takes a number of pictures, saving them to a folder created. This can be done many times.
The user then clicks 'Train Images'. The system does so and notifies you when it's done. The user can then take attendance of all students with live video. Note that a person's attendance is only documented once per session, no matter how many times the user is recognized. Click 'q' to close the camera window. 

The attendance is saved to a csv file which can be viewed on Microsoft excel. 
For this project, I used an external camera (phone) and connected it using an application called DroidCam. But if you wish to use the webcam of your system, change the video cam number from 1 to 0 in the code.

Run AttendanceFI.py to try the software.

Also, I got help from a repository by @ashishdubey10 and multiple sources on google. :)


