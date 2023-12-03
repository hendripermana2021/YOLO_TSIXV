import cv2
import pickle
import cvzone
import numpy as np
import datetime
import time
import streamlit as st
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import tempfile
import os

# File uploader and video display
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
cap = None

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    uploaded_file.seek(0)

    # Open the video file with cv2.VideoCapture
    cap = cv2.VideoCapture(temp_file.name)

    # YOLOv5 initialization
    net = cv2.dnn.readNetFromONNX("C:\\Users\\M Fathurrahman\\Documents\\computer-vision-SParking (1)\\computer-vision-SParking\\best (4).onnx")
    classes = ['mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil']
    
    # Video output setup
    frameSize = [852, 480]
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "Video\Hasil/"
    timestamp_video = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(video_path + f"Video_{timestamp_video}.mp4", cv2_fourcc, cap.get(cv2.CAP_PROP_FPS) - 10, frameSize)

    # Load parking positions from a file
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)


# Function to check parking space
def checkParkingSpace(imgPro, car_boxes):  # Fungsion local
    width, height = 15, 30
    spaceCounter = 0
    counter = 1
    free_slots = []

    for i, pos in enumerate(posList):
        x, y = pos
        found = False
        for car_box in car_boxes:
            car_x, car_y, car_w, car_h = car_box
            if car_x < x + width and car_x + car_w > x and car_y < y + height and car_y + car_h > y:
                color = (0, 0, 255)
                thickness = 1
                found = True
                break
        if not found:
            color = (0, 255, 0)
            thickness = 1
            spaceCounter += 1
            free_slots.append(i + 1)
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cv2.putText(img, str(counter), (pos[0]+5, pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        counter += 1
        cvzone.putTextRect(img, str(counter), (x, y + height - 3), scale=1,
                        thickness=1, offset=0, colorR=color)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cvzone.putTextRect (img, f'Time: {timestamp}', (10, 25), scale = 0.5, thickness = 1, offset = 10, colorR = (10,10,10)) 
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (10, 50), scale=0.5, thickness=1, offset=10, colorR=(0,200,225))
    cvzone.putTextRect(img, f'Slot: {str(free_slots)}', (200, 25), scale=0.5, thickness=1, offset=10, colorR=(0,150,0)) 
    cvzone.putTextRect(img, f'Cars: {len(posList)-spaceCounter}', (200, 50), scale=0.5, thickness=1, offset=10, colorR=(102,7,0)) 


while cap is not None:
    success, img = cap.read()
    if img is None:
        break

    # Resize the frame
    img = cv2.resize(img, (852, 480))

    # YOLOv5 detection
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]


    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 1)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    car_boxes = []
    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.2:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                car_boxes.append(box)

    checkParkingSpace(imgDilate, car_boxes)

    indices = cv2.dnn.NMSBoxes(car_boxes, confidences, 0.2, 0.2)

    for i in indices:
        x1, y1, w, h = car_boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 255), 1)

    time.sleep(1)
    out.write(img)

    # Display the processed frame in Streamlit
    st.image(img, channels="BGR", use_column_width=True)

