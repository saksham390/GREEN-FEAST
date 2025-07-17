import cv2
import os
import time
from twilio.rest import Client
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6  # Lower is more strict
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

# Twilio Setup (replace with your credentials)
TWILIO_SID = "AC1f22ae3dd9af5573f71e8dd567434a41"  # Your Twilio SID
TWILIO_TOKEN = "78500f750c6cf777fb4960d97792f887"  # Your Twilio token
TWILIO_NUMBER = "+12524659622"  # Your Twilio phone number

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Initialize Twilio client
sms_client = Client(TWILIO_SID, TWILIO_TOKEN)

def extract_face_embedding(image):
    """Extract face embedding using MediaPipe Face Mesh"""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    
    # Convert landmarks to a numpy array
    landmarks = np.array([[lm.x, lm.y, lm.z] 
                         for lm in results.multi_face_landmarks[0].landmark])
    
    # Simple embedding - can be replaced with more sophisticated methods
    embedding = landmarks.flatten()
    return embedding

def load_known_faces():
    known_embeddings = []
    known_names = []
    known_phones = []
    
    print("🔍 Loading known faces...")
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        # Load phone number
        phone_file = os.path.join(person_dir, "phone.txt")
        if not os.path.exists(phone_file):
            print(f"⚠️ No phone number for {person_name}")
            continue
            
        with open(phone_file, "r") as f:
            phone_number = f.read().strip()
        
        # Load all images of the person
        for filename in os.listdir(person_dir):
            if filename == "phone.txt":
                continue
                
            image_path = os.path.join(person_dir, filename)
            image = cv2.imread(image_path)
            
            embedding = extract_face_embedding(image)
            if embedding is not None:
                known_embeddings.append(embedding)
                known_names.append(person_name)
                known_phones.append(phone_number)
            else:
                print(f"❌ No face found in {filename}")
    
    return known_embeddings, known_names, known_phones

def send_sms(phone, name):
    try:
        message = sms_client.messages.create(
            body=f"👋 Hello {name}! Your face was just recognized at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            from_=TWILIO_NUMBER,
            to=phone
        )
        print(f"📤 SMS sent to {name} at {phone}")
    except Exception as e:
        print(f"❌ SMS failed: {str(e)}")

def main():
    known_embeddings, known_names, known_phones = load_known_faces()
    video = cv2.VideoCapture(0)
    
    # Track last notification time for each person
    last_notification = {}
    
    print("🎥 Starting face recognition...")
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                left = int(bboxC.xmin * iw)
                top = int(bboxC.ymin * ih)
                right = left + int(bboxC.width * iw)
                bottom = top + int(bboxC.height * ih)
                
                # Extract face region
                face_region = frame[top:bottom, left:right]
                
                # Get face embedding
                face_embedding = extract_face_embedding(face_region)
                if face_embedding is None:
                    continue
                
                # Compare with known faces
                name = "Unknown"
                phone = None
                
                if known_embeddings:
                    # Calculate cosine similarity
                    similarities = []
                    for known_embedding in known_embeddings:
                        # Reshape to 2D arrays for cosine_similarity
                        sim = cosine_similarity(
                            face_embedding.reshape(1, -1),
                            known_embedding.reshape(1, -1)
                        )[0][0]
                        similarities.append(sim)
                    
                    best_match_idx = np.argmax(similarities)
                    best_similarity = similarities[best_match_idx]
                    
                    if best_similarity > (1 - TOLERANCE):
                        name = known_names[best_match_idx]
                        phone = known_phones[best_match_idx]
                        
                        # Send SMS if not sent in last 5 minutes
                        current_time = time.time()
                        if name not in last_notification or current_time - last_notification[name] > 300:
                            send_sms(phone, name)
                            last_notification[name] = current_time
                
                # Draw rectangle and label
                color = [0, 255, 0] if name != "Unknown" else [0, 0, 255]
                cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), FONT_THICKNESS)
                
                if phone:
                    cv2.putText(frame, phone, (left + 6, bottom + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Face Recognition', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
    face_detection.close()
    face_mesh.close()

if __name__ == "__main__":
    main()
