import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO('D:/VsCode Projects/Helmet-And-Number-Plate-detection-RealTIme/runs/detect/train/weights/best.pt')
    return model

model = load_model()

# Function to process an image and detect objects
def detect_objects(image):
    results = model(image)
    return results

# Streamlit app layout
st.title("Helmet and Number Plate Detection App")
st.write("Upload an image or use your webcam for real-time helmet and number plate detection.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    results = detect_objects(image)
    
    # Process results
    annotated_image = results[0].plot()  # Get the annotated image from the first result
    st.image(annotated_image, caption='Processed Image', use_column_width=True)
    
    # Display detection results
    detections = results[0].boxes.xyxy.numpy()  # Get bounding box coordinates
    for det in detections:
        class_id = int(det[-1])  # Class ID
        confidence = det[-2]  # Confidence score
        st.write(f"Detected Class ID: {class_id} with confidence: {confidence:.2f}")

# Upload video
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(video_path)
    frame_window = st.empty()  # Create an empty frame in Streamlit
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(frame)
        annotated_frame = results[0].plot()  # Get the annotated image from the first result

        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update the image displayed in Streamlit
        frame_window.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()
    import os
    os.remove(video_path)  # Optionally delete the temporary video file

# Initialize session state for webcam control
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

# Real-time video detection
if st.button('Start Webcam'):
    st.session_state.run_webcam = True

if st.button('Stop Webcam'):
    st.session_state.run_webcam = False

# Run webcam detection if the flag is set
if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0)
    frame_window = st.empty()  # Create an empty frame in Streamlit

    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        results = detect_objects(frame)
        annotated_frame = results[0].plot()  # Get the annotated image from the first result

        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update the image displayed in Streamlit
        frame_window.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()