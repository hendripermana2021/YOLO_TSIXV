import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import pickle
import datetime
# import gspread

cap = cv2.VideoCapture("Video/Smart Parking_T.mp4")  # For Video
log = {}
slot_status = {}
masuk_slot = {}
keluar_slot = {}
waktu_mobil = {}
prev_slot_status = {1:False,2:False, 3:False, 4:False, 5:False, 6:False, 7:False, 8:False}

model = YOLO("yolov8.pt")

classNames = ["mobil"]
# frameSize = [852,480]
# cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_path = "Video\Hasil/"
# timestamp_video = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# out = cv2.VideoWriter(video_path+f"Video_{timestamp_video}.mp4", cv2_fourcc, cap.get(cv2.CAP_PROP_FPS)-10, frameSize)
# screnShoot = cv2.VideoCapture(0)
duration = 10

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)
   
    scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
    ]

# Fungsi ID mobil
def cariIdMobildariIDParkir(idParkir):
    for jik, cva in log.items():
        if cva == idParkir:
            return int(jik)
    return "E"

# Funngsi mengecek slot parkir
def checkParkingSpace(imgPro, car_boxes):  # Fungsion local
    log_per_frame = {}
    width, height = 15, 30
    spaceCounter = 0
    free_slots = []
    for i, pos in enumerate(posList):
        x, y = pos


        found = False
        current_id = ""
        for car_box in car_boxes:
            car_x, car_y, car_w, car_h, car_id = car_box
            if car_x < x + width and car_x + car_w > x and car_y < y + height and car_y + car_h > y:
                color = (0, 0, 255)
                thickness = 1
                found = True
                slot_status.update({(i+1):True})
                if not (car_id in log):
                    log_value = {car_id:(i+1)}
                    log.update(log_value)
                break
        
        if not found:
            slot_status.update({(i+1):False})
            color = (0, 255, 0)
            thickness = 1
            spaceCounter += 1
            free_slots.append(i + 1)
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(i+1), (x, y + height - 3), scale=1,
                        thickness=1, offset=0, colorR=color)


    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cvzone.putTextRect (img, f'Time: {timestamp}', (10, 25), scale = 1, thickness = 1, offset = 10, colorR = (10,10,10)) 
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (10, 60), scale=1, thickness= 1, offset=10, colorR=(0,200,225))
    cvzone.putTextRect(img, f'Slot: {str(free_slots)}', (400, 25), scale=0.5, thickness= 1, offset=10, colorR=(0,150,0)) 
    cvzone.putTextRect(img, f'Cars: {len(posList)-spaceCounter}', (400, 60), scale=1, thickness= 1, offset=10, colorR=(102,7,0)) 
    # credentials = ServiceAccountCredentials.from_json_keyfile_name("Smart_Parking.json", scopes) #access the json key you downloaded earlier 
    # file = gspread.authorize(credentials) # authenticate the JSON key with gspread
    # sheet = file.open("Smart_Parking") #open sheet
    # sheet = sheet.sheet1 #replace sheet_name with the name that corresponds to yours, e.g, it can be sheet1
    # row = [timestamp, f'Free: {spaceCounter}/{len(posList)}', str(free_slots)]
    # index = 3
    # sheet.insert_row(row, index)

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []

while True:
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")

    # Runing model
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img, (852,480))
    results = model(img, stream=True)
    detections = np.empty((0, 5))
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
    carboxes=[]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections) 

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        carbox=np.array([x1, y1, w, h, id])
        carboxes.append(carbox)

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=1, offset=5)
     
        checkParkingSpace(imgDilate, carboxes)
    n = 0
    for k, v in slot_status.items():
        sebelumnya =prev_slot_status[k]
        if sebelumnya!=v:
            # Masuk
            if v == True and sebelumnya == False:
                masuk_slot.update({k:now})
                if k in keluar_slot:
                    keluar_slot.pop(k)
            # Mobil Keluar
            if v == False and sebelumnya == True:
                keluar_slot.update({k:now})
                idMobil = cariIdMobildariIDParkir(k)
                if idMobil in waktu_mobil:
                   waktu = waktu_mobil[idMobil] + waktu; 
                waktu_mobil.update({idMobil: waktu})
                log.pop(idMobil)
    
    # Fungsi menghitung rentang waktu parkir
    for s in range(1,9):
        if s in keluar_slot:
            waktu = keluar_slot[s]-masuk_slot[s]
        else:
            waktu = datetime.timedelta(seconds=0)
            if s in masuk_slot:
                waktu = now-masuk_slot[s]
        waktuShow = str(waktu)
        if "." in waktuShow:
            waktuShow,milis = (waktuShow).split(".")
        cv2.putText(img, f"ID {s}", (int(852-170), int(40+(n*20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(img, f"{cariIdMobildariIDParkir(s)}", (int(852-125), int(40+(n*20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,0,255), 1)
        cv2.putText(img, f"{waktuShow}", (int(852-100), int(40+(n*20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        n += 1

    # Lama waktu mobil parkir
    print(waktu_mobil)

    cv2.imshow("Image", img)
    prev_slot_status = dict(slot_status)
    p = cv2.waitKey(1)
    if p == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break