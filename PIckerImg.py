import cv2
import pickle
import streamlit as st
 


width, height = 15, 30
 
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# tambahkan variabel global untuk drag kotak parkir
dragParkir = False
parkirTerpilih = None
offsetX, offsetY = 0, 0
 
def mouseClick(events, x, y, flags, params):
    global posList, dragParkir, parkirTerpilih, offsetX, offsetY

    if events == cv2.EVENT_LBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                parkirTerpilih = i
                dragParkir = True
                offsetX = x - x1
                offsetY = y - y1

        if not dragParkir:
            posList.append((x, y))

    if events == cv2.EVENT_LBUTTONUP:
        dragParkir = False

    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y1 + height:
                posList.pop(i)

    if dragParkir and parkirTerpilih is not None:
        posList[parkirTerpilih] = (x - offsetX, y - offsetY)

    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)
 
while True:
    img = cv2.imread('Foto\Smart Parking.png')
    counter = 1
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 1)
        cv2.putText(img, str(counter), (pos[0]+5, pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        counter += 1

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)

    # menambahkan fitur hapus kotak dengan menekan tombol 'x'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        posList.clear()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
