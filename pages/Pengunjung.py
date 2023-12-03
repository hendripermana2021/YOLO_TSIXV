import cv2
import pickle
import cvzone
import numpy as np
import datetime
import streamlit as st
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import tempfile
import os

st.set_page_config(
    page_title="Aplikasi Pengunjung",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
img = None

if uploaded_file is not None:
    # Read the uploaded image
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # YOLOv5 initialization
    net = cv2.dnn.readNetFromONNX("C:\\Users\\M Fathurrahman\\Documents\\computer-vision-SParking (1)\\computer-vision-SParking\\TSixV2.onnx")
    classes = ['mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil', 'mobil']

    # Load parking positions from a file
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)

    # Resize the image
    img = cv2.resize(img, (852, 480))

    # YOLOv5 detection
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    classes_ids = []
    confidences = []
    car_boxes = []

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

    for i in range(detections.shape[0]):
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
                box = np.array([x1, y1, width, height])
                car_boxes.append(box)

    # Function to check parking space
    def checkParkingSpace(imgPro, car_boxes):
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

        # Display information below the image
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        st.title("Parking Space Information")
        st.text(f'Time: {timestamp}')
        st.text(f'Free Spaces: {spaceCounter}/{len(posList)}')
        st.text(f'Occupied Spaces: {len(posList)-spaceCounter}')
        st.text(f'Slots: {str(free_slots)}')

        # Display metrics in Streamlit columns
        a1, a2 = st.columns((2, 2))

        with a1:
            st.markdown(f'<div class="metric-container">Ketersediaan Parkir:<br><span class="metric-label">{spaceCounter}/{len(posList)}</span></div>', unsafe_allow_html=True)
        with a2:
            st.markdown(f'<div class="metric-container">Slot:<br><span class="metric-label">{str(free_slots)}</span></div>', unsafe_allow_html=True)


    checkParkingSpace(imgDilate, car_boxes)

    indices = cv2.dnn.NMSBoxes(car_boxes, confidences, 0.2, 0.2)

    for i in indices:
        x1, y1, w, h = car_boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 255), 1)

    # Display the processed image in Streamlit
    st.image(img, channels="BGR", use_column_width=True)
    with open('style2.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
