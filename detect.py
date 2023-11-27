
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

# Rest of the code remains the same...

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
haar_data =  cv2.CascadeClassifier(cascPathface)
with_mask = np.load('400mask.npy')
without_mask = np.load('400Nomask.npy')
with_mask = with_mask.reshape(400,50 * 50 * 3)
without_mask = without_mask.reshape(400,50 * 50 * 3)
# print (with_mask.shape)
# print(without_mask.shape)

X = np.r_[with_mask, without_mask]
# print (X.shape)
labels = np.zeros(X.shape[0])

labels[400:] = 1.0

names = {0: 'mask', 1: 'no mask'}

#sklearn - scikit-learn library of machine learning

x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.20)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
# print(x_train.shape)
# print (x_train[0])

# Eigen Value / Eigne vector
# print(x_train.shape)
svm = SVC()
svm.fit(x_train, y_train)
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)
print(accuracy_score(y_test, y_pred))

# ... (previous code)

# ... (previous code)

# Function to update the label with mask detection result and accuracy
def update_label(text):
    label.config(text=text)
    
accuracy_values = []
time_values = []

# Function to start face mask detection
def start_detection():
    global detection_running
    detection_running = True
    capture = cv2.VideoCapture(0)
    data = []
    font = cv2.FONT_HERSHEY_COMPLEX
    while detection_running:
        flag, img = capture.read()
        if flag:
            
            faces = haar_data.detectMultiScale(img)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
                face = img[y:y + h, x:x + w, :]
                face = cv2.resize(face, (50, 50)) 
                face = face.reshape(1, -1)
                face = pca.transform(face)
                pred = svm.predict(face)
                n = names[int( pred)]
                accuracy = accuracy_score([1 - y_test[0]], [1 - pred]) * 100
                # accuracy = accuracy_score(y_test, y_pred)
                # text = f"{n} {accuracy:.2f}%"
                text = f"{n}"
                
                # accuracy = accuracy_score([1 - y_test[0]], [1 - pred]) * 100
                accuracy_values.append(accuracy)
                time_values.append(time.time())
                plot_graph()
                time.sleep(0)
                # text = f"{n}%"
                
                cv2.putText(img, text, (x, y), font, 1, (244, 250, 250), 2)
                if len(data) < 400:
                    data.append(face)
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.imgtk = imgtk
        root.update()
    capture.release()

# ... (rest of the code)

# ... (rest of the code)

def plot_graph():
    # plt.figure(figsize=(10, 5))
    plt.plot(time_values, accuracy_values)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Time')
    plt.show()


# Function to stop face mask detection
def stop_detection():
    global detection_running
    detection_running = False

# Main window
root = tk.Tk()
root.title("Face Mask Detection User Dashboard")

# Custom Style for "Start" button
style_start = ttk.Style()
style_start.configure("Start.TButton", background="blue", foreground="blue",
                      borderwidth=0, focuscolor="none", focusthickness=0, highlightthickness=0, relief="flat",
                      font=("Helvetica", 14), padx=15, pady=5, borderradius=5)

# Custom Style for "Stop" button
style_stop = ttk.Style()
style_stop.configure("Stop.TButton", background="red", foreground="red",
                     borderwidth=0, focuscolor="none", focusthickness=0, highlightthickness=0, relief="flat",
                     font=("Helvetica", 14), padx=15, pady=5, borderradius=5)

# Add a label for the title
title_label = tk.Label(root, text="Face Mask Detection User Dashboard", font=("Helvetica", 20), pady=20, fg='#fff', bg='#0a79f7', width=96)
title_label.pack()

# Add a canvas to display webcam feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack(pady=10)

# Add a label to show face mask detection result
label = tk.Label(root, text="", font=("Helvetica", 16))
label.pack(pady=10)

# Add a frame to center the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Add the "Start" button to start face mask detection
start_button = ttk.Button(button_frame, text="Start", style="Start.TButton",
                          command=lambda: threading.Thread(target=start_detection).start())
start_button.pack(side=tk.LEFT, padx=10)

# Add the "Stop" button to stop face mask detection
stop_button = ttk.Button(button_frame, text="Stop", style="Stop.TButton", command=stop_detection)
stop_button.pack(side=tk.LEFT, padx=10)

root.mainloop()

