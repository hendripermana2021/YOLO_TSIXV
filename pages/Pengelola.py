import cv2
import pickle
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Aplikasi Pengelola",
    layout="wide",
    initial_sidebar_state="expanded"
)
width, height = 15, 30
with open('pages/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Counter for box enumeration
box_counter = 1

# Function to draw boxes on the image
def draw_boxes(image, boxes):
    for i, pos in enumerate(boxes):
        cv2.rectangle(image, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 1)
        cv2.putText(image, str(i + 1), (pos[0] + 5, pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Function to draw guide lines
def draw_guide_lines(image, x, y):
    cv2.line(image, (0, y), (image.shape[1], y), (0, 255, 0), 1)  # Horizontal line
    cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 1)  # Vertical line

# Function to update the position list
def update_pos_list(x, y):
    global posList
    posList.append((x, y))

def save_pos_list(pos_list):
    with open('shared_data.txt', 'w') as f:
        for pos in pos_list:
            f.write(f"{pos[0]},{pos[1]}\n")

# Create Streamlit app
st.title("Smart Parking System")

uploaded_file = st.file_uploader("Choose an image...", type=('jpg', 'jpeg', 'png', 'gif'))
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Clear boxes button
    if st.button("Clear Boxes (Press 'x')"):
        posList.clear()

    # Image display
    st.image(img, use_column_width=True, channels="BGR", caption="Uploaded Image")

    # Sliders for X and Y coordinates
    x_slider = st.slider("X Coordinate", 0, img.shape[1] - 1, img.shape[1] // 2)
    y_slider = st.slider("Y Coordinate", 0, img.shape[0] - 1, img.shape[0] // 2)

    # Button to add box
    if st.button("Add Box"):
        update_pos_list(x_slider, y_slider)

    # Drawing guide lines
    draw_guide_lines(img, x_slider, y_slider)

    # Drawing boxes on the image
    draw_boxes(img, posList)
    st.image(img, use_column_width=True, channels="BGR", caption="Uploaded Image")

    # Display enumeration of boxes
    st.write("Boxes:")
    for i, pos in enumerate(posList):
        st.write(f"Box {i + 1}: {pos}")

    # Save posList to file
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)
    save_pos_list(posList)

    # Save the image bytes to a shared file
    with open('shared_image.jpg', 'wb') as f:
        f.write(file_bytes)