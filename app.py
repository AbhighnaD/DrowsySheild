from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import threading
import pygame
import math
import os
import time

app = Flask(__name__)

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Correct eye landmark indices for EAR calculation
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Simplified for EAR calculation
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Simplified for EAR calculation

# Mouth landmarks for MAR calculation
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]

# Initialize pygame mixer for alarm
pygame.mixer.init()

# Global variables
detection_data = {
    'ear': 0.25,
    'mar': 0.3,
    'eyes_closed': False,
    'yawning': False,
    'closed_frames': 0,
    'yawn_frames': 0,
    'alarm_triggered': False,
    'fatigue_warning': False
}

stats = {
    'eyes_closed_count': 0,
    'yawning_count': 0,
    'total_alarms': 0
}

# Detection parameters
EAR_THRESHOLD = 0.20  # Eye aspect ratio threshold
MAR_THRESHOLD = 0.75  # Mouth aspect ratio threshold
CONSECUTIVE_FRAMES = 20  # Frames needed to trigger alarm

# Thread control
is_detecting = False
cap = None
alarm_playing = False
warning_playing = False

def calculate_ear(eye_points, landmarks, img_w, img_h):
    """Calculate Eye Aspect Ratio"""
    try:
        # Extract the specific eye landmarks
        p2 = landmarks[eye_points[1]]  # Top
        p3 = landmarks[eye_points[2]]  # Top
        p5 = landmarks[eye_points[4]]  # Bottom
        p6 = landmarks[eye_points[5]]  # Bottom
        p1 = landmarks[eye_points[0]]  # Left corner
        p4 = landmarks[eye_points[3]]  # Right corner
        
        # Calculate vertical distances
        vert1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
        vert2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
        
        # Calculate horizontal distance
        horiz = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)
        
        if horiz == 0:
            return 0.25
            
        ear = (vert1 + vert2) / (2.0 * horiz)
        return ear
    except Exception as e:
        print(f"EAR calculation error: {e}")
        return 0.25

def calculate_mar(landmarks, img_w, img_h):
    """Calculate Mouth Aspect Ratio"""
    try:
        # Use inner mouth points for better accuracy
        left = landmarks[78]
        right = landmarks[308]
        top = landmarks[13]
        bottom = landmarks[14]
        
        # Calculate mouth width and height
        mouth_width = math.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
        mouth_height = math.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
        
        if mouth_width == 0:
            return 0.3
            
        mar = mouth_height / mouth_width
        return mar
    except Exception as e:
        print(f"MAR calculation error: {e}")
        return 0.3

def play_alarm():
    """Play alarm sound from static folder"""
    global alarm_playing
    try:
        if not alarm_playing:
            alarm_playing = True
            # Try to play the alarm sound from static folder
            alarm_path = os.path.join('static', 'alarm.mp3')
            print(f"Looking for alarm at: {alarm_path}")
            if os.path.exists(alarm_path):
                print("Alarm file found, playing sound...")
                pygame.mixer.music.load(alarm_path)
                pygame.mixer.music.play()
                # Wait for sound to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                print("Alarm sound finished!")
            else:
                print(f"Alarm file not found: {alarm_path}")
                # Fallback: use system beep if available
                try:
                    import winsound
                    winsound.Beep(1000, 1000)
                except:
                    print("No fallback sound available")
            alarm_playing = False
    except Exception as e:
        print(f"Error playing alarm: {e}")
        alarm_playing = False

def play_warning():
    """Play warning sound from static folder"""
    global warning_playing
    try:
        if not warning_playing:
            warning_playing = True
            # Try to play the warning sound from static folder
            warning_path = os.path.join('static', 'warning.mp3')
            print(f"Looking for warning at: {warning_path}")
            if os.path.exists(warning_path):
                print("Warning file found, playing sound...")
                sound = pygame.mixer.Sound(warning_path)
                sound.play()
                # Wait for sound to finish (adjust time based on your sound file)
                time.sleep(2)
                print("Warning sound finished!")
            else:
                print(f"Warning file not found: {warning_path}")
                # Fallback: use system beep if available
                try:
                    import winsound
                    winsound.Beep(500, 500)
                except:
                    print("No fallback sound available")
            warning_playing = False
    except Exception as e:
        print(f"Error playing warning: {e}")
        warning_playing = False

def generate_frames():
    """Generate video frames with real-time detection"""
    global is_detecting, detection_data, stats, cap
    
    frame_count = 0
    
    while is_detecting and cap is not None:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Calculate EAR and MAR
                    left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks, w, h)
                    right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    mar = calculate_mar(landmarks, w, h)
                    
                    # Update detection data
                    detection_data['ear'] = round(avg_ear, 3)
                    detection_data['mar'] = round(mar, 3)
                    
                    # Debug print every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"EAR: {avg_ear:.3f}, MAR: {mar:.3f}")
                    
                    # Eye closure detection
                    if avg_ear < EAR_THRESHOLD:
                        detection_data['closed_frames'] += 1
                        detection_data['eyes_closed'] = True
                        
                        if detection_data['closed_frames'] >= CONSECUTIVE_FRAMES and not detection_data['alarm_triggered']:
                            detection_data['alarm_triggered'] = True
                            stats['eyes_closed_count'] += 1
                            stats['total_alarms'] += 1
                            print("ALARM TRIGGERED: Eyes closed for too long!")
                            # Play alarm sound in a separate thread
                            if not alarm_playing:
                                threading.Thread(target=play_alarm, daemon=True).start()
                    else:
                        detection_data['closed_frames'] = 0
                        detection_data['eyes_closed'] = False
                        detection_data['alarm_triggered'] = False

                    # Yawning detection
                    if mar > MAR_THRESHOLD:
                        detection_data['yawn_frames'] += 1
                        detection_data['yawning'] = True
                        
                        # Count yawn after sustained detection
                        if detection_data['yawn_frames'] == 15:
                            stats['yawning_count'] += 1
                            print("Yawning detected!")
                            # Play warning sound for yawning
                            if not warning_playing:
                                threading.Thread(target=play_warning, daemon=True).start()
                        
                        # Show fatigue warning for prolonged yawning
                        if detection_data['yawn_frames'] > 30:
                            detection_data['fatigue_warning'] = True
                    else:
                        detection_data['yawn_frames'] = 0
                        detection_data['yawning'] = False
                        detection_data['fatigue_warning'] = False

                    # Add overlay to frame
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    if detection_data['eyes_closed']:
                        cv2.putText(frame, "EYES CLOSED!", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if detection_data['yawning']:
                        cv2.putText(frame, "YAWNING DETECTED!", (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    if detection_data['alarm_triggered']:
                        cv2.putText(frame, "DROWSINESS ALERT!", (w//2-150, h//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    if detection_data['fatigue_warning']:
                        cv2.putText(frame, "FATIGUE WARNING!", (w//2-150, h//2-50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

                    # Display frame counts
                    cv2.putText(frame, f"Closed Frames: {detection_data['closed_frames']}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Yawn Frames: {detection_data['yawn_frames']}", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            else:
                # No face detected - reset counters
                detection_data['ear'] = 0.0
                detection_data['mar'] = 0.0
                detection_data['closed_frames'] = 0
                detection_data['yawn_frames'] = 0
                detection_data['eyes_closed'] = False
                detection_data['yawning'] = False
                detection_data['alarm_triggered'] = False
                detection_data['fatigue_warning'] = False

                cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display counters on frame
            cv2.putText(frame, f"Eyes Alarms: {stats['eyes_closed_count']}", (w-300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Alarms: {stats['total_alarms']}", (w-300, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Yawn Count: {stats['yawning_count']}", (w-300, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation: {e}")
            time.sleep(0.1)
            continue

@app.route('/')
def index():
    return render_template('drowsiness_detector.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def get_detection_data():
    """Return current detection data as JSON"""
    return jsonify({
        'detection': detection_data,
        'stats': stats
    })

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start the drowsiness detection"""
    global is_detecting, cap
    
    if not is_detecting:
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({'status': 'error', 'message': 'Cannot open camera'})
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            is_detecting = True
            print("Detection started - camera initialized")
            
            return jsonify({'status': 'started', 'message': 'Detection started successfully'})
        except Exception as e:
            print(f"Error starting detection: {e}")
            if cap:
                cap.release()
                cap = None
            return jsonify({'status': 'error', 'message': f'Error starting detection: {str(e)}'})
    else:
        return jsonify({'status': 'already_running', 'message': 'Detection already running'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop the drowsiness detection"""
    global is_detecting, cap
    
    if is_detecting:
        is_detecting = False
        
        # Stop camera
        if cap:
            cap.release()
            cap = None
        
        # Stop any playing sounds
        pygame.mixer.music.stop()
        pygame.mixer.stop()
        
        # Reset detection data
        detection_data.update({
            'ear': 0.25,
            'mar': 0.3,
            'eyes_closed': False,
            'yawning': False,
            'closed_frames': 0,
            'yawn_frames': 0,
            'alarm_triggered': False,
            'fatigue_warning': False
        })
        
        print("Detection stopped")
        return jsonify({'status': 'stopped', 'message': 'Detection stopped'})
    else:
        return jsonify({'status': 'not_running', 'message': 'Detection not running'})

@app.route('/reset_counters', methods=['POST'])
def reset_counters():
    """Reset all counters and detection data"""
    global detection_data, stats
    
    detection_data.update({
        'ear': 0.25,
        'mar': 0.3,
        'eyes_closed': False,
        'yawning': False,
        'closed_frames': 0,
        'yawn_frames': 0,
        'alarm_triggered': False,
        'fatigue_warning': False
    })
    
    stats.update({
        'eyes_closed_count': 0,
        'yawning_count': 0,
        'total_alarms': 0
    })
    
    # Stop any playing sounds
    pygame.mixer.music.stop()
    pygame.mixer.stop()
    
    print("Counters reset")
    return jsonify({'status': 'reset', 'message': 'Counters reset'})

if __name__ == '__main__':
    print("Starting Drowsiness Detection Server...")
    print("Make sure you have 'alarm.mp3' and 'warning.mp3' in the static folder")
    print("Access the web interface at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)