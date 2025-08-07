from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')  # Make sure yolov8s.pt is in your current directory or specify the full path

# Train the model
model.train(data='C:/Users/Aaman/OneDrive/Desktop/Helmet-And-Number-Plate-detection-RealTIme/custom_dataset/coco128.yaml', 
            epochs=50, 
            imgsz=640, 
            device='cpu')  # Change '0' to 'cpu' if you're not using a GPU
