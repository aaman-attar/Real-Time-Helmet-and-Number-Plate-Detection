# 🛵 Helmet and Number Plate Detection - Real-Time 🚔

This project is a real-time detection system for:
- Detecting **riders without helmets**
- Detecting **number plates** of vehicles

Built using:
- **YOLOv8** for object detection
- **Streamlit** for web interface
- **OpenCV** for real-time webcam video feed
- **Python** as the primary language

---

## 📂 Project Structure

Helmet-And-Number-Plate-Detection-RealTime/
│
├── app.py # Main Streamlit app
├── detect_helmet.py # Helmet detection module
├── detect_number_plate.py # Number plate detection module
├── yolov8_helmet.pt # Trained YOLOv8 model for helmets
├── yolov8_number_plate.pt # Trained YOLOv8 model for number plates
├── requirements.txt
├── README.md
├── venv virtual environment 


---

## 🚀 How to Run

1. **Clone the repository**:

bash
git clone https://github.com/your-username/Helmet-And-Number-Plate-Detection-RealTime.git
cd Helmet-And-Number-Plate-Detection-RealTime


2. Create virtual environment (optional but recommended):

python -m venv venv
venv\Scripts\activate     # For Windows
# source venv/bin/activate   # For Linux/macOS


3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit App:

streamlit run app.py

Features
📷 Real-time webcam detection

🧠 YOLOv8-based object detection

🎯 Detects:

Motorcyclists without helmets

Vehicle number plates

🌐 Simple web interface using Streamlit
