import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

class PupilMouseControl:
    def __init__(self):
        # Initialize camera and face mesh
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Get screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Safety settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
        
        # Pupil tracking parameters
        self.pupil_history = []
        self.smoothing_factor = 8  # Higher for more stability
        self.sensitivity = 2.5  # Mouse sensitivity multiplier
        
        # Calibration system
        self.calibrated = False
        self.calibration_points = []
        self.calibration_stage = 0
        self.calibration_data = []
        self.calibration_timer = 0
        
        # Blink detection for clicking
        self.ear_threshold = 0.2
        self.blink_frames = 0
        self.min_blink_frames = 3
        self.max_blink_frames = 12
        self.last_click_time = 0
        self.click_cooldown = 1.0  # Longer cooldown for safety
        
        # Dwell clicking (alternative to blink)
        self.dwell_enabled = True
        self.dwell_time = 2.0  # seconds to dwell for click
        self.dwell_start_time = 0
        self.dwell_position = None
        self.dwell_threshold = 30  # pixels movement threshold
        
        # Eye region detection
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Mouse boundaries (safety margins)
        self.margin_x = 50
        self.margin_y = 50
        
        print("👁️ Pupil Mouse Control for Paralyzed Users")
        print("Features:")
        print("1. High-precision pupil tracking")
        print("2. Automatic calibration system")
        print("3. Blink-to-click OR dwell-to-click")
        print("4. Smooth movement with noise reduction")
        print("5. Safety boundaries")
        print("\nControls:")
        print("- Look around to move cursor")
        print("- Blink deliberately to click")
        print("- OR hold gaze steady for 2 seconds to click")
        print("- Press 'q' to quit")
        print("- Press 'r' to recalibrate")
        print("- Press 'd' to toggle dwell clicking")
        
    def extract_pupil_position(self, landmarks, frame, eye_landmarks):
        """Extract pupil position from eye region"""
        try:
            # Get eye region coordinates
            eye_points = []
            for landmark_id in eye_landmarks:
                x = int(landmarks[landmark_id].x * frame.shape[1])
                y = int(landmarks[landmark_id].y * frame.shape[0])
                eye_points.append([x, y])
            
            eye_points = np.array(eye_points)
            
            # Create bounding box around eye
            x_min, y_min = np.min(eye_points, axis=0)
            x_max, y_max = np.max(eye_points, axis=0)
            
            # Add padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)
            
            # Extract eye region
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            if eye_region.shape[0] < 10 or eye_region.shape[1] < 10:
                return None
            
            # Convert to grayscale
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
            
            # Find the darkest point (pupil)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
            
            # Convert back to frame coordinates
            pupil_x = min_loc[0] + x_min
            pupil_y = min_loc[1] + y_min
            
            # Normalize to eye region for better tracking
            eye_center_x = (x_min + x_max) // 2
            eye_center_y = (y_min + y_max) // 2
            
            # Calculate relative position
            rel_x = (pupil_x - eye_center_x) / ((x_max - x_min) / 2)
            rel_y = (pupil_y - eye_center_y) / ((y_max - y_min) / 2)
            
            return {
                'absolute': (pupil_x, pupil_y),
                'relative': (rel_x, rel_y),
                'eye_center': (eye_center_x, eye_center_y),
                'eye_region': (x_min, y_min, x_max, y_max)
            }
            
        except Exception as e:
            print(f"Error extracting pupil: {e}")
            return None
    
    def calculate_gaze_direction(self, left_pupil, right_pupil):
        """Calculate gaze direction from both pupils"""
        if left_pupil is None or right_pupil is None:
            return None
        
        # Average the relative positions
        avg_x = (left_pupil['relative'][0] + right_pupil['relative'][0]) / 2
        avg_y = (left_pupil['relative'][1] + right_pupil['relative'][1]) / 2
        
        return (avg_x, avg_y)
    
    def smooth_movement(self, x, y):
        """Apply smoothing to reduce jitter"""
        self.pupil_history.append((x, y))
        
        if len(self.pupil_history) > self.smoothing_factor:
            self.pupil_history.pop(0)
        
        if len(self.pupil_history) > 1:
            # Weighted average with more weight on recent positions
            weights = np.linspace(0.1, 1.0, len(self.pupil_history))
            weights = weights / np.sum(weights)
            
            avg_x = np.average([pos[0] for pos in self.pupil_history], weights=weights)
            avg_y = np.average([pos[1] for pos in self.pupil_history], weights=weights)
            
            return int(avg_x), int(avg_y)
        
        return int(x), int(y)
    
    def calibrate_system(self):
        """5-point calibration system"""
        # Define calibration points (corners + center)
        self.calibration_points = [
            (self.screen_w // 4, self.screen_h // 4),           # Top-left
            (3 * self.screen_w // 4, self.screen_h // 4),       # Top-right
            (self.screen_w // 2, self.screen_h // 2),           # Center
            (self.screen_w // 4, 3 * self.screen_h // 4),       # Bottom-left
            (3 * self.screen_w // 4, 3 * self.screen_h // 4)    # Bottom-right
        ]
        
        self.calibration_stage = 0
        self.calibration_data = []
        self.calibrated = False
        print("🎯 Starting calibration. Look at the red circles.")
    
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Left eye EAR
        left_A = math.sqrt((landmarks[159].x - landmarks[145].x)**2 + (landmarks[159].y - landmarks[145].y)**2)
        left_B = math.sqrt((landmarks[158].x - landmarks[153].x)**2 + (landmarks[158].y - landmarks[153].y)**2)
        left_C = math.sqrt((landmarks[33].x - landmarks[133].x)**2 + (landmarks[33].y - landmarks[133].y)**2)
        
        # Right eye EAR
        right_A = math.sqrt((landmarks[386].x - landmarks[374].x)**2 + (landmarks[386].y - landmarks[374].y)**2)
        right_B = math.sqrt((landmarks[385].x - landmarks[380].x)**2 + (landmarks[385].y - landmarks[380].y)**2)
        right_C = math.sqrt((landmarks[362].x - landmarks[263].x)**2 + (landmarks[362].y - landmarks[263].y)**2)
        
        # Calculate EAR
        left_ear = (left_A + left_B) / (2.0 * left_C) if left_C > 0 else 0.25
        right_ear = (right_A + right_B) / (2.0 * right_C) if right_C > 0 else 0.25
        
        return (left_ear + right_ear) / 2.0
    
    def handle_dwell_clicking(self, mouse_x, mouse_y):
        """Handle dwell-based clicking"""
        if not self.dwell_enabled:
            return False
        
        current_time = time.time()
        
        if self.dwell_position is None:
            self.dwell_position = (mouse_x, mouse_y)
            self.dwell_start_time = current_time
            return False
        
        # Calculate distance from initial dwell position
        distance = math.sqrt((mouse_x - self.dwell_position[0])**2 + 
                           (mouse_y - self.dwell_position[1])**2)
        
        if distance > self.dwell_threshold:
            # Reset dwell if moved too far
            self.dwell_position = (mouse_x, mouse_y)
            self.dwell_start_time = current_time
            return False
        
        # Check if dwell time exceeded
        if current_time - self.dwell_start_time >= self.dwell_time:
            if current_time - self.last_click_time > self.click_cooldown:
                self.dwell_position = None
                return True
        
        return False
    
    def run(self):
        """Main loop"""
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        # Extract pupil positions
                        left_pupil = self.extract_pupil_position(landmarks, frame, self.left_eye_landmarks)
                        right_pupil = self.extract_pupil_position(landmarks, frame, self.right_eye_landmarks)
                        
                        # Calculate gaze direction
                        gaze = self.calculate_gaze_direction(left_pupil, right_pupil)
                        
                        if gaze is not None:
                            # Map gaze to screen coordinates
                            gaze_x, gaze_y = gaze
                            
                            # Apply sensitivity and invert X for natural movement
                            mouse_x = self.screen_w // 2 - int(gaze_x * self.screen_w * self.sensitivity)
                            mouse_y = self.screen_h // 2 + int(gaze_y * self.screen_h * self.sensitivity)
                            
                            # Apply boundaries
                            mouse_x = max(self.margin_x, min(self.screen_w - self.margin_x, mouse_x))
                            mouse_y = max(self.margin_y, min(self.screen_h - self.margin_y, mouse_y))
                            
                            # Smooth movement
                            smooth_x, smooth_y = self.smooth_movement(mouse_x, mouse_y)
                            
                            # Move mouse
                            try:
                                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                            except pyautogui.FailSafeException:
                                print("Emergency stop activated!")
                                break
                            
                            # Handle clicking
                            current_time = time.time()
                            clicked = False
                            
                            # Blink detection
                            ear = self.calculate_ear(landmarks)
                            
                            if ear < self.ear_threshold:
                                self.blink_frames += 1
                            else:
                                if (self.min_blink_frames <= self.blink_frames <= self.max_blink_frames and
                                    current_time - self.last_click_time > self.click_cooldown):
                                    clicked = True
                                    print(f"🖱️ BLINK CLICK! (EAR: {ear:.3f})")
                                self.blink_frames = 0
                            
                            # Dwell clicking
                            if not clicked and self.handle_dwell_clicking(smooth_x, smooth_y):
                                clicked = True
                                print("🖱️ DWELL CLICK!")
                            
                            # Perform click
                            if clicked:
                                pyautogui.click()
                                self.last_click_time = current_time
                                cv2.circle(frame, (50, 50), 25, (0, 255, 0), -1)
                            
                            # Display info
                            cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Mouse: ({smooth_x}, {smooth_y})", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Dwell indicator
                            if self.dwell_enabled and self.dwell_position is not None:
                                dwell_progress = (current_time - self.dwell_start_time) / self.dwell_time
                                if dwell_progress < 1.0:
                                    cv2.putText(frame, f"Dwell: {dwell_progress:.1%}", (10, 120), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Draw pupil positions
                            if left_pupil:
                                cv2.circle(frame, left_pupil['absolute'], 3, (0, 0, 255), -1)
                                # Draw eye region
                                x1, y1, x2, y2 = left_pupil['eye_region']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            
                            if right_pupil:
                                cv2.circle(frame, right_pupil['absolute'], 3, (0, 0, 255), -1)
                                # Draw eye region
                                x1, y1, x2, y2 = right_pupil['eye_region']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                # Display the frame
                cv2.imshow("Pupil Mouse Control", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.calibrate_system()
                elif key == ord('d'):
                    self.dwell_enabled = not self.dwell_enabled
                    print(f"Dwell clicking: {'ENABLED' if self.dwell_enabled else 'DISABLED'}")
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cam.release()
        cv2.destroyAllWindows()
        print("👋 Pupil Mouse Control stopped")

def main():
    """Main function"""
    try:
        controller = PupilMouseControl()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
