import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class EyeMouseControl:
    def __init__(self):
        # Initialize camera and face mesh
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Get screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Safety settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
        
        # Blink detection parameters
        self.ear_threshold = 0.2  # Will be calibrated
        self.blink_frames = 0
        self.min_blink_frames = 2
        self.max_blink_frames = 8
        self.last_click_time = 0
        self.click_cooldown = 0.8  # seconds between clicks
        
        # Calibration
        self.ear_history = []
        self.calibrated = False
        self.calibration_frames = 0
        
        # Mouse smoothing
        self.mouse_history = []
        self.smoothing_factor = 5
        
        print("🎯 Eye Mouse Control with Blink Click")
        print("Instructions:")
        print("1. Position your face clearly in front of the camera")
        print("2. Look normally for 3 seconds (calibration)")
        print("3. Move your eyes to move the mouse cursor")
        print("4. Blink deliberately to click")
        print("5. Press 'q' to quit")
        print("6. Move mouse to top-left corner to emergency stop")
        
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Left eye landmarks (6 points)
        left_eye = [
            landmarks[33],   # outer corner
            landmarks[160],  # top
            landmarks[158],  # top
            landmarks[133],  # inner corner
            landmarks[153],  # bottom
            landmarks[144]   # bottom
        ]
        
        # Right eye landmarks (6 points)
        right_eye = [
            landmarks[362],  # outer corner
            landmarks[385],  # top
            landmarks[387],  # top
            landmarks[263],  # inner corner
            landmarks[373],  # bottom
            landmarks[380]   # bottom
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
            
            # Calculate EAR
            if C > 0:
                return (A + B) / (2.0 * C)
            return 0.3
        
        # Calculate EAR for both eyes and average
        left_ear = get_ear(left_eye)
        right_ear = get_ear(right_eye)
        
        return (left_ear + right_ear) / 2.0
    
    def get_gaze_position(self, landmarks):
        """Calculate gaze position for mouse control"""
        # Use multiple eye landmarks for better stability
        left_eye_center = np.mean([
            [landmarks[33].x, landmarks[33].y],
            [landmarks[133].x, landmarks[133].y],
            [landmarks[160].x, landmarks[160].y],
            [landmarks[144].x, landmarks[144].y]
        ], axis=0)
        
        right_eye_center = np.mean([
            [landmarks[362].x, landmarks[362].y],
            [landmarks[263].x, landmarks[263].y],
            [landmarks[385].x, landmarks[385].y],
            [landmarks[380].x, landmarks[380].y]
        ], axis=0)
        
        # Average both eyes
        gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
        gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
        
        return gaze_x, gaze_y
    
    def smooth_mouse_movement(self, x, y):
        """Smooth mouse movement to reduce jitter"""
        self.mouse_history.append((x, y))
        
        if len(self.mouse_history) > self.smoothing_factor:
            self.mouse_history.pop(0)
        
        # Calculate smoothed position
        if len(self.mouse_history) > 1:
            avg_x = sum(pos[0] for pos in self.mouse_history) / len(self.mouse_history)
            avg_y = sum(pos[1] for pos in self.mouse_history) / len(self.mouse_history)
            return int(avg_x), int(avg_y)
        
        return int(x), int(y)
    
    def run(self):
        """Main loop"""
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        # Calculate EAR for blink detection
                        ear = self.calculate_ear(landmarks)
                        
                        # Calibration phase
                        if not self.calibrated:
                            self.calibration_frames += 1
                            self.ear_history.append(ear)
                            
                            cv2.putText(frame, f"CALIBRATING... {max(0, 90 - self.calibration_frames)}", 
                                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            if self.calibration_frames >= 90:  # 3 seconds at 30fps
                                # Calculate threshold from calibration data
                                if len(self.ear_history) > 0:
                                    avg_ear = np.mean(self.ear_history)
                                    self.ear_threshold = avg_ear * 0.7  # 70% of normal EAR
                                    self.calibrated = True
                                    print(f"✅ Calibration complete! Blink threshold: {self.ear_threshold:.3f}")
                                else:
                                    self.calibration_frames = 0
                        
                        else:
                            # Main tracking phase
                            
                            # Get gaze position for mouse control
                            gaze_x, gaze_y = self.get_gaze_position(landmarks)
                            
                            # Map gaze position to screen coordinates
                            # Add some margin and invert coordinates for natural movement
                            mouse_x = int((1 - gaze_x) * self.screen_w)
                            mouse_y = int(gaze_y * self.screen_h)
                            
                            # Smooth mouse movement
                            smooth_x, smooth_y = self.smooth_mouse_movement(mouse_x, mouse_y)
                            
                            # Move mouse
                            try:
                                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                            except pyautogui.FailSafeException:
                                print("Emergency stop activated!")
                                break
                            
                            # Blink detection for clicking
                            current_time = time.time()
                            
                            if ear < self.ear_threshold:
                                self.blink_frames += 1
                            else:
                                if (self.min_blink_frames <= self.blink_frames <= self.max_blink_frames and
                                    current_time - self.last_click_time > self.click_cooldown):
                                    
                                    # Perform click
                                    pyautogui.click()
                                    self.last_click_time = current_time
                                    print(f"🖱️ CLICK! (EAR: {ear:.3f})")
                                    
                                    # Visual feedback
                                    cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
                                
                                self.blink_frames = 0
                            
                            # Display info on frame
                            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Mouse: ({smooth_x}, {smooth_y})", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Blink Frames: {self.blink_frames}", (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Draw eye landmarks for visual feedback
                            for landmark_id in [33, 133, 160, 144, 362, 263, 385, 380]:
                                x = int(landmarks[landmark_id].x * frame_w)
                                y = int(landmarks[landmark_id].y * frame_h)
                                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Display the frame
                cv2.imshow("Eye Mouse Control", frame)
                
                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cam.release()
        cv2.destroyAllWindows()
        print("👋 Eye Mouse Control stopped")

def main():
    """Main function"""
    try:
        controller = EyeMouseControl()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
