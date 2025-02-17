import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Use forward slashes or escape the backslashes in file paths
faceCascade = cv2.CascadeClassifier('C:/Users/luvvg/OneDrive/Desktop/Project Work/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/Users/luvvg/OneDrive/Desktop/Project Work/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

# Create the main window
root = tk.Tk()
root.title("Eye Tracking Program")

# Create a label to display the video feed
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# Variable to track whether the video feed is running
video_running = False

# Variable to track the number of detected faces
num_faces = 0

# Create a label to display the number of detected faces
num_faces_label = tk.Label(root, text="Number of Faces: 0")
num_faces_label.pack(pady=10)

def update_video():
    global num_faces
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    num_faces = len(faces)
    num_faces_label.config(text=f"Number of Faces: {num_faces}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    video_label.img = img
    video_label.config(image=img)

    if video_running:
        video_label.after(10, update_video)

# Function to start the eye tracking program
def start_eye_tracking():
    global video_running
    video_running = True
    update_video()

# Function to stop/pause the eye tracking program
def stop_eye_tracking():
    global video_running
    video_running = False

# Create a Start button
start_button = tk.Button(root, text="Start", command=start_eye_tracking)
start_button.pack(side=tk.LEFT, padx=5)

# Create a Stop button
stop_button = tk.Button(root, text="Stop", command=stop_eye_tracking)
stop_button.pack(side=tk.LEFT, padx=5)

# Create an exit button to close the program
exit_button = tk.Button(root, text="Exit", command=root.destroy)
exit_button.pack(side=tk.RIGHT, padx=5)

root.mainloop()

# Release the video capture and close all windows when the GUI is closed
cap.release()
cv2.destroyAllWindows()
