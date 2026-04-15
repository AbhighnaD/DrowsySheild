import cv2
import mediapipe as mp
import time
import threading
import pygame
import math

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices for EAR calculation
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth landmark indices for MAR calculation (correct indices)
MOUTH_UPPER = [13, 82, 80, 81, 312, 311, 310, 415]  # Upper lip points
MOUTH_LOWER = [14, 87, 88, 89, 317, 402, 318, 324]  # Lower lip points
MOUTH_CORNERS = [78, 308]  # Mouth corners

# Initialize pygame mixer for alarm
pygame.mixer.init()

# Counters
eyes_closed_count = 0
alarm_trigger_count = 0
yawning_count = 0

def play_alarm():
    try:
        pygame.mixer.music.load(r"C:\Users\Adnan Ahmed\Desktop\DDD\DD\drowsy-env\alarm.mp3")
        pygame.mixer.music.play()
        print("Alarm triggered!")
    except Exception as e:
        print(f"Error playing alarm: {e}")
        import winsound
        winsound.Beep(1000, 1000)

def calculate_ear(eye_points, landmarks, img_w, img_h):
    """Calculate Eye Aspect Ratio"""
    # Extract the 6 key points for EAR calculation
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]] 
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]
    
    # Calculate Euclidean distances
    vertical1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
    vertical2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
    horizontal = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)
    
    # Avoid division by zero
    if horizontal == 0:
        return 0.0
        
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def calculate_mar(landmarks, img_w, img_h):
    """Calculate Mouth Aspect Ratio - CORRECTED VERSION"""
    # Get mouth corner points
    left_corner = landmarks[78]    # Left mouth corner
    right_corner = landmarks[308]  # Right mouth corner
    
    # Get upper and lower lip points
    upper_lip = landmarks[13]      # Upper lip center
    lower_lip = landmarks[14]      # Lower lip center
    
    # Calculate mouth width (horizontal distance between corners)
    mouth_width = math.sqrt((left_corner.x - right_corner.x)**2 + 
                           (left_corner.y - right_corner.y)**2)
    
    # Calculate mouth height (vertical distance between upper and lower lip)
    mouth_height = math.sqrt((upper_lip.x - lower_lip.x)**2 + 
                            (upper_lip.y - lower_lip.y)**2)
    
    # Avoid division by zero
    if mouth_width == 0:
        return 0.0
        
    # Mouth Aspect Ratio = height / width
    mar = mouth_height / mouth_width
    return mar

# Start webcam
cap = cv2.VideoCapture(0)

CLOSED_EYES_FRAME = 0
ALARM_ON = False
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75  # Correct threshold for yawning (0.7-0.8 for open mouth)
CONSECUTIVE_FRAMES = 30
YAWNING_FRAMES = 0

print("Starting Drowsiness Detection...")
print("Press 'ESC' to quit, 'r' to reset counters")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR for yawning detection
            mar = calculate_mar(landmarks, w, h)
            
            # Display metrics
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Eye closure detection
            if avg_ear < EAR_THRESHOLD:
                CLOSED_EYES_FRAME += 1
                cv2.putText(frame, "EYES CLOSED!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                CLOSED_EYES_FRAME = 0
                if ALARM_ON:
                    ALARM_ON = False
                    pygame.mixer.music.stop()

            # Yawning detection - CORRECTED
            if mar > MAR_THRESHOLD:
                YAWNING_FRAMES += 1
                cv2.putText(frame, "YAWNING DETECTED!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Count yawn only after sustained detection
                if YAWNING_FRAMES == 10:  # Count after 10 frames of continuous yawning
                    yawning_count += 1
                    
                # Show fatigue warning for prolonged yawning
                if YAWNING_FRAMES > 25:
                    cv2.putText(frame, "FATIGUE WARNING!", (w//2-150, h//2-50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
            else:
                YAWNING_FRAMES = 0

            # Display current frame counts
            cv2.putText(frame, f"Closed Frames: {CLOSED_EYES_FRAME}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Yawn Frames: {YAWNING_FRAMES}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Trigger alarm for prolonged eye closure
            if CLOSED_EYES_FRAME >= CONSECUTIVE_FRAMES and not ALARM_ON:
                ALARM_ON = True
                eyes_closed_count += 1
                alarm_trigger_count += 1
                threading.Thread(target=play_alarm, daemon=True).start()
                cv2.putText(frame, "DROWSINESS ALERT!", (w//2-150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Display all counters
    cv2.putText(frame, f"Eyes Closed Alarms: {eyes_closed_count}", (w-300, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Alarms: {alarm_trigger_count}", (w-300, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Yawning Count: {yawning_count}", (w-300, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset counters
        eyes_closed_count = 0
        alarm_trigger_count = 0
        yawning_count = 0
        CLOSED_EYES_FRAME = 0
        ALARM_ON = False
        pygame.mixer.music.stop()
        print("Counters reset!")

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()