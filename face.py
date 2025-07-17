import cv2
import numpy as np
import os  # Added missing import
import time
import qrcode
from twilio.rest import Client
import mediapipe as mp
from ultralytics import YOLO

# ======== Configuration ========
# Face Recognition
KNOWN_FACES_DIR = "known_faces"
FACE_TOLERANCE = 0.6

# Food Detection
FOOD_DENSITY = {"rice": 0.02, "roti": 0.03, "curry": 0.015, "dal": 0.01}
FINE_RATE = 0.10  # ₹0.10 per gram
MIN_WASTE_THRESHOLD = 50  # Minimum 50g waste to trigger fine

# Twilio Setup (replace with your credentials)
TWILIO_SID = "AC1f22ae3dd9af5573f71e8dd567434a41"
TWILIO_TOKEN = "78500f750c6cf777fb4960d97792f887"
TWILIO_NUMBER = "+12524659622"

# ======== Initialize Systems ========
# MediaPipe for Face Recognition
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# YOLO for Food Detection
food_model = YOLO("yolov8n.pt")  # Replace with your trained model

# Twilio Client
sms_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Video Capture
cap = cv2.VideoCapture(0)

# ======== Core Functions ========
def load_known_faces():
    """Load known faces from directory"""
    known_faces = []
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        return known_faces

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        phone_file = os.path.join(person_dir, "phone.txt")
        if not os.path.exists(phone_file):
            continue
            
        with open(phone_file, "r") as f:
            phone = f.read().strip()
        
        # Load sample face images
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    known_faces.append({
                        "name": person_name,
                        "phone": phone,
                        "image": image
                    })
    return known_faces

def detect_faces(frame):
    """Detect faces in the frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    
    recognized = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw = frame.shape[:2]
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                          int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Ensure valid face region
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= iw and y+h <= ih:
                face_region = frame[y:y+h, x:x+w]
                if face_region.size > 0:
                    recognized.append({
                        "name": "shivam",  # Simplified - replace with actual recognition
                        "phone": "+917050525501",  # Default for demo
                        "location": (x, y, w, h)
                    })
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, "Guest", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame, recognized

def calculate_waste(current_frame, initial_weights):
    """Calculate wasted food"""
    results = food_model(current_frame)
    waste_data = {}
    total_fine = 0
    
    for result in results:
        for box in result.boxes:
            food_class = food_model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_weight = (x2-x1)*(y2-y1)*FOOD_DENSITY.get(food_class, 0.01)
            
            wasted = max(0, initial_weights.get(food_class, 0) - current_weight)
            if wasted > 0:
                fine = wasted * FINE_RATE
                total_fine += fine
                waste_data[food_class] = wasted
                
                # Visual warning
                cv2.putText(current_frame, f"WASTED: {wasted:.1f}g", 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0,0,255), 2)
    
    return current_frame, waste_data, total_fine

def send_alert(phone, name, fine_amount):
    """Send SMS and generate QR code"""
    upi_link = f"upi://pay?pa=foodwaste@upi&am={fine_amount:.2f}&tn=WasteFine"
    qr = qrcode.make(upi_link)
    qr_path = f"fine_{fine_amount:.2f}.png"
    qr.save(qr_path)
    
    try:
        sms_client.messages.create(
            body=f"🚨 Food Waste Alert!\nDear {name}, please pay ₹{fine_amount:.2f} fine\nPay: {upi_link}",
            from_=TWILIO_NUMBER,
            to=phone
        )
        print(f"📨 SMS sent to {name} at {phone}")
    except Exception as e:
        print(f"❌ SMS failed: {e}")
    
    return qr_path

# ======== Main Loop ========
def main():
    known_faces = load_known_faces()
    initial_weights = {}
    last_notification = {}
    scanning_initial = True
    
    print("🚀 System starting...")
    print("1. Show full thali to camera for initial scan")
    print("2. System will automatically detect waste")
    print("3. Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Face detection
        frame, recognized_people = detect_faces(frame)
        
        # 2. Food analysis
        if scanning_initial:
            # Initial scan mode
            results = food_model(frame)
            if results:
                initial_weights = {
                    food_model.names[int(box.cls)]: (int(box.xyxy[0][2])-int(box.xyxy[0][0]))*(int(box.xyxy[0][3])-int(box.xyxy[0][1]))*FOOD_DENSITY.get(food_model.names[int(box.cls)], 0.01)
                    for result in results for box in result.boxes
                }
                if initial_weights:
                    print("✅ Initial thali scanned. Weights:", initial_weights)
                    scanning_initial = False
        else:
            # Waste monitoring mode
            frame, waste_data, total_fine = calculate_waste(frame, initial_weights)
            
            # Trigger alert if waste exceeds threshold
            if total_fine >= MIN_WASTE_THRESHOLD and recognized_people:
                current_time = time.time()
                for person in recognized_people:
                    if (person["name"] not in last_notification or 
                        current_time - last_notification[person["name"]] > 300):
                        
                        qr_path = send_alert(person["phone"], person["name"], total_fine)
                        last_notification[person["name"]] = current_time
                        
                        # Show QR code
                        qr_img = cv2.imread(qr_path)
                        if qr_img is not None:
                            cv2.imshow("Payment QR", qr_img)
        
        # Display status
        status = "Initial Scan" if scanning_initial else "Waste Monitoring"
        cv2.putText(frame, f"Mode: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Smart Thali Monitoring", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System stopped")

if __name__ == "__main__":
    main()
