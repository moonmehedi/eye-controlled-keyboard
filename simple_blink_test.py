import cv2
import mediapipe as mp
import numpy as np
import time

class SimpleBlink:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        
        # Blink detection
        self.ear_threshold = 0.25
        self.blink_frames = 0
        self.min_blink_frames = 3
        self.max_blink_frames = 15
        self.last_blink_time = 0
        self.blink_cooldown = 1.0
        
        # Calibration
        self.ear_history = []
        self.calibrated = False
        self.calibration_frames = 0
        
        print("🎯 Simple Blink Detection Test")
        print("Instructions:")
        print("1. Look at camera normally for 5 seconds (calibration)")
        print("2. Then blink deliberately to test detection")
        print("3. Press 'q' to quit")
        
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio"""
        # Left eye landmarks
        left_eye = [
            landmarks[33], landmarks[160], landmarks[158], 
            landmarks[133], landmarks[153], landmarks[144]
        ]
        
        # Right eye landmarks
        right_eye = [
            landmarks[362], landmarks[385], landmarks[387], 
            landmarks[263], landmarks[373], landmarks[380]
        ]
        
        def get_ear(eye_points):
            # Vertical distances
            A = np.sqrt((eye_points[1].x - eye_points[5].x)**2 + 
                       (eye_points[1].y - eye_points[5].y)**2)
            B = np.sqrt((eye_points[2].x - eye_points[4].x)**2 + 
                       (eye_points[2].y - eye_points[4].y)**2)
            # Horizontal distance
            C = np.sqrt((eye_points[0].x - eye_points[3].x)**2 + 
                       (eye_points[0].y - eye_points[3].y)**2)
            
            return (A + B) / (2.0 * C)
        
        left_ear = get_ear(left_eye)
        right_ear = get_ear(right_eye)
        
        return (left_ear + right_ear) / 2.0
    
    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Calculate EAR
                    ear = self.calculate_ear(landmarks)
                    
                    # Calibration phase
                    if not self.calibrated:
                        self.calibration_frames += 1
                        self.ear_history.append(ear)
                        
                        if self.calibration_frames < 150:  # 5 seconds
                            cv2.putText(frame, f"CALIBRATING... {150 - self.calibration_frames}", 
                                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            # Calculate threshold
                            avg_ear = np.mean(self.ear_history)
                            self.ear_threshold = avg_ear * 0.75
                            self.calibrated = True
                            print(f"✅ Calibrated! Average EAR: {avg_ear:.3f}, Threshold: {self.ear_threshold:.3f}")
                    
                    # Blink detection
                    if self.calibrated:
                        # Display current values
                        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, f"Threshold: {self.ear_threshold:.3f}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, f"Blink Frames: {self.blink_frames}", (10, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Check for blink
                        if ear < self.ear_threshold:
                            self.blink_frames += 1
                            cv2.putText(frame, "BLINKING!", (10, 170), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            # Check if valid blink occurred
                            if self.min_blink_frames <= self.blink_frames <= self.max_blink_frames:
                                current_time = time.time()
                                if current_time - self.last_blink_time > self.blink_cooldown:
                                    print(f"🎯 BLINK DETECTED! Frames: {self.blink_frames}")
                                    self.last_blink_time = current_time
                                    
                                    # Flash screen
                                    cv2.rectangle(frame, (0, 0), (640, 480), (0, 255, 0), 20)
                                    cv2.putText(frame, "BLINK DETECTED!", (150, 250), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            
                            self.blink_frames = 0
                    
                    # Draw eye landmarks
                    for i in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
                        x = int(landmarks[i].x * frame.shape[1])
                        y = int(landmarks[i].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            cv2.imshow('Simple Blink Test - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SimpleBlink()
    app.run()
