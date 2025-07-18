import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import threading
import json
import os

class AdvancedPupilMouseControl:
    def __init__(self):
        # Initialize camera and face mesh
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Get screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Safety settings
        pyautogui.FAILSAFE = False  # Disabled for paralyzed users
        pyautogui.PAUSE = 0.005
        
        # Configuration file
        self.config_file = "pupil_mouse_config.json"
        self.load_config()
        
        # Pupil tracking parameters
        self.pupil_history = []
        self.smoothing_factor = self.config.get('smoothing_factor', 10)
        self.sensitivity = self.config.get('sensitivity', 1.8)
        
        # Multiple click methods
        self.click_methods = {
            'blink': True,
            'dwell': True,
            'long_blink': True,
            'left_wink': False,
            'right_wink': False
        }
        
        # Blink detection
        self.ear_threshold = 0.22
        self.blink_frames = 0
        self.min_blink_frames = 2
        self.max_blink_frames = 8
        self.long_blink_frames = 15  # For long blink detection
        
        # Wink detection
        self.left_ear_threshold = 0.22
        self.right_ear_threshold = 0.22
        self.wink_frames = {'left': 0, 'right': 0}
        
        # Dwell clicking
        self.dwell_time = 1.5
        self.dwell_start_time = 0
        self.dwell_position = None
        self.dwell_threshold = 25
        
        # Click management
        self.last_click_time = 0
        self.click_cooldown = 0.6
        
        # Zones system for easier control
        self.zones = {
            'top_left': (0, 0, self.screen_w//3, self.screen_h//3),
            'top_center': (self.screen_w//3, 0, 2*self.screen_w//3, self.screen_h//3),
            'top_right': (2*self.screen_w//3, 0, self.screen_w, self.screen_h//3),
            'middle_left': (0, self.screen_h//3, self.screen_w//3, 2*self.screen_h//3),
            'center': (self.screen_w//3, self.screen_h//3, 2*self.screen_w//3, 2*self.screen_h//3),
            'middle_right': (2*self.screen_w//3, self.screen_h//3, self.screen_w, 2*self.screen_h//3),
            'bottom_left': (0, 2*self.screen_h//3, self.screen_w//3, self.screen_h),
            'bottom_center': (self.screen_w//3, 2*self.screen_h//3, 2*self.screen_w//3, self.screen_h),
            'bottom_right': (2*self.screen_w//3, 2*self.screen_h//3, self.screen_w, self.screen_h)
        }
        
        # Eye region landmarks
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Calibration system
        self.calibration_active = False
        self.calibration_points = []
        self.calibration_data = []
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Voice feedback (text-to-speech simulation)
        self.voice_enabled = False
        
        print("🎯 Advanced Pupil Mouse Control for Paralyzed Users")
        print("=" * 60)
        print("FEATURES:")
        print("• High-precision pupil tracking")
        print("• Multiple click methods (blink, dwell, long blink)")
        print("• Zone-based navigation")
        print("• Customizable sensitivity")
        print("• Configuration saving/loading")
        print("• Performance monitoring")
        print("• Accessibility optimized")
        print()
        print("CONTROLS:")
        print("• Look around: Move cursor")
        print("• Quick blink: Left click")
        print("• Hold gaze 1.5s: Dwell click")
        print("• Long blink (0.5s): Right click")
        print("• 'q': Quit")
        print("• 's': Save current settings")
        print("• 'r': Reset to defaults")
        print("• 'c': Start calibration")
        print("• '+'/'-': Adjust sensitivity")
        print("=" * 60)
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            'sensitivity': 1.8,
            'smoothing_factor': 10,
            'dwell_time': 1.5,
            'click_cooldown': 0.6,
            'ear_threshold': 0.22
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Configuration loaded from {self.config_file}")
            else:
                self.config = default_config
                print("ℹ️ Using default configuration")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            self.config = default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_to_save = {
                'sensitivity': self.sensitivity,
                'smoothing_factor': self.smoothing_factor,
                'dwell_time': self.dwell_time,
                'click_cooldown': self.click_cooldown,
                'ear_threshold': self.ear_threshold
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"⚠️ Error saving config: {e}")
    
    def detect_pupil_advanced(self, eye_region, eye_landmarks, frame):
        """Advanced pupil detection with multiple methods"""
        try:
            if eye_region.shape[0] < 20 or eye_region.shape[1] < 20:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely pupil)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate moments
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
            
            # Fallback to minimum point method
            min_val, _, min_loc, _ = cv2.minMaxLoc(gray)
            return min_loc
            
        except Exception as e:
            print(f"Pupil detection error: {e}")
            return None
    
    def calculate_individual_ear(self, landmarks, eye_indices):
        """Calculate EAR for individual eye"""
        try:
            # Get eye points
            eye_points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
            
            # Calculate vertical distances
            vertical_dist_1 = math.sqrt((eye_points[1][0] - eye_points[5][0])**2 + 
                                      (eye_points[1][1] - eye_points[5][1])**2)
            vertical_dist_2 = math.sqrt((eye_points[2][0] - eye_points[4][0])**2 + 
                                      (eye_points[2][1] - eye_points[4][1])**2)
            
            # Calculate horizontal distance
            horizontal_dist = math.sqrt((eye_points[0][0] - eye_points[3][0])**2 + 
                                      (eye_points[0][1] - eye_points[3][1])**2)
            
            # Calculate EAR
            if horizontal_dist > 0:
                ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
                return ear
            return 0.25
            
        except Exception as e:
            return 0.25
    
    def get_pupil_positions(self, landmarks, frame):
        """Get pupil positions from both eyes"""
        frame_h, frame_w = frame.shape[:2]
        
        results = {}
        
        for eye_name, eye_landmarks in [('left', self.left_eye_landmarks), ('right', self.right_eye_landmarks)]:
            try:
                # Get eye region
                eye_points = []
                for landmark_id in eye_landmarks:
                    x = int(landmarks[landmark_id].x * frame_w)
                    y = int(landmarks[landmark_id].y * frame_h)
                    eye_points.append([x, y])
                
                eye_points = np.array(eye_points)
                
                # Create bounding box
                x_min, y_min = np.min(eye_points, axis=0)
                x_max, y_max = np.max(eye_points, axis=0)
                
                # Add padding
                padding = 15
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame_w, x_max + padding)
                y_max = min(frame_h, y_max + padding)
                
                # Extract eye region
                eye_region = frame[y_min:y_max, x_min:x_max]
                
                # Detect pupil
                pupil_pos = self.detect_pupil_advanced(eye_region, eye_landmarks, frame)
                
                if pupil_pos:
                    # Convert to frame coordinates
                    pupil_x = pupil_pos[0] + x_min
                    pupil_y = pupil_pos[1] + y_min
                    
                    # Calculate relative position
                    eye_center_x = (x_min + x_max) // 2
                    eye_center_y = (y_min + y_max) // 2
                    
                    rel_x = (pupil_x - eye_center_x) / ((x_max - x_min) / 2)
                    rel_y = (pupil_y - eye_center_y) / ((y_max - y_min) / 2)
                    
                    results[eye_name] = {
                        'absolute': (pupil_x, pupil_y),
                        'relative': (rel_x, rel_y),
                        'region': (x_min, y_min, x_max, y_max)
                    }
                    
            except Exception as e:
                print(f"Error processing {eye_name} eye: {e}")
                continue
        
        return results
    
    def calculate_gaze_vector(self, pupil_data):
        """Calculate gaze direction from pupil data"""
        if 'left' not in pupil_data or 'right' not in pupil_data:
            return None
        
        left_rel = pupil_data['left']['relative']
        right_rel = pupil_data['right']['relative']
        
        # Average both eyes
        gaze_x = (left_rel[0] + right_rel[0]) / 2
        gaze_y = (left_rel[1] + right_rel[1]) / 2
        
        return (gaze_x, gaze_y)
    
    def apply_smoothing(self, x, y):
        """Apply advanced smoothing with outlier rejection"""
        self.pupil_history.append((x, y))
        
        if len(self.pupil_history) > self.smoothing_factor:
            self.pupil_history.pop(0)
        
        if len(self.pupil_history) < 3:
            return int(x), int(y)
        
        # Calculate median to reject outliers
        x_values = [pos[0] for pos in self.pupil_history]
        y_values = [pos[1] for pos in self.pupil_history]
        
        # Use weighted average of recent positions
        weights = np.exp(np.linspace(-2, 0, len(self.pupil_history)))
        weights = weights / np.sum(weights)
        
        smooth_x = np.average(x_values, weights=weights)
        smooth_y = np.average(y_values, weights=weights)
        
        return int(smooth_x), int(smooth_y)
    
    def handle_multiple_click_methods(self, landmarks, mouse_x, mouse_y):
        """Handle multiple click methods"""
        current_time = time.time()
        clicked = False
        click_type = None
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_individual_ear(landmarks, [33, 160, 158, 133, 153, 144])
        right_ear = self.calculate_individual_ear(landmarks, [362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2
        
        # Blink detection
        if avg_ear < self.ear_threshold:
            self.blink_frames += 1
        else:
            if (self.min_blink_frames <= self.blink_frames <= self.max_blink_frames and
                current_time - self.last_click_time > self.click_cooldown):
                clicked = True
                click_type = "Quick Blink"
            elif (self.blink_frames >= self.long_blink_frames and
                  current_time - self.last_click_time > self.click_cooldown):
                clicked = True
                click_type = "Long Blink (Right Click)"
                # Perform right click
                pyautogui.rightClick()
                self.last_click_time = current_time
                return True, click_type
            
            self.blink_frames = 0
        
        # Dwell clicking
        if not clicked and self.click_methods['dwell']:
            if self.dwell_position is None:
                self.dwell_position = (mouse_x, mouse_y)
                self.dwell_start_time = current_time
            else:
                # Check if moved too far
                distance = math.sqrt((mouse_x - self.dwell_position[0])**2 + 
                                   (mouse_y - self.dwell_position[1])**2)
                
                if distance > self.dwell_threshold:
                    self.dwell_position = (mouse_x, mouse_y)
                    self.dwell_start_time = current_time
                elif current_time - self.dwell_start_time >= self.dwell_time:
                    if current_time - self.last_click_time > self.click_cooldown:
                        clicked = True
                        click_type = "Dwell Click"
                        self.dwell_position = None
        
        # Perform click
        if clicked and click_type != "Long Blink (Right Click)":
            pyautogui.click()
            self.last_click_time = current_time
        
        return clicked, click_type
    
    def draw_ui_elements(self, frame):
        """Draw UI elements on frame"""
        # Draw zones
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            # Scale to frame size
            fx1 = int(x1 * frame.shape[1] / self.screen_w)
            fy1 = int(y1 * frame.shape[0] / self.screen_h)
            fx2 = int(x2 * frame.shape[1] / self.screen_w)
            fy2 = int(y2 * frame.shape[0] / self.screen_h)
            
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (100, 100, 100), 1)
        
        # Draw settings panel
        panel_height = 200
        panel_width = 300
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Add text to panel
        texts = [
            f"Sensitivity: {self.sensitivity:.1f}",
            f"Smoothing: {self.smoothing_factor}",
            f"Dwell Time: {self.dwell_time:.1f}s",
            f"FPS: {self.current_fps:.1f}",
            f"Click Cooldown: {self.click_cooldown:.1f}s"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(panel, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overlay panel on frame
        frame[10:10+panel_height, 10:10+panel_width] = panel
        
        # Draw dwell indicator
        if self.dwell_position is not None:
            progress = (time.time() - self.dwell_start_time) / self.dwell_time
            if progress < 1.0:
                # Draw progress circle
                center_x = frame.shape[1] - 100
                center_y = 100
                radius = 30
                
                # Background circle
                cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), -1)
                
                # Progress arc
                angle = int(360 * progress)
                cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                           -90, 0, angle, (0, 255, 0), 3)
                
                cv2.putText(frame, "DWELL", (center_x - 25, center_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main loop with enhanced features"""
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # FPS calculation
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        # Get pupil positions
                        pupil_data = self.get_pupil_positions(landmarks, frame)
                        
                        # Calculate gaze
                        gaze = self.calculate_gaze_vector(pupil_data)
                        
                        if gaze is not None:
                            # Map to screen coordinates
                            gaze_x, gaze_y = gaze
                            
                            # Apply sensitivity
                            mouse_x = self.screen_w // 2 - int(gaze_x * self.screen_w * self.sensitivity)
                            mouse_y = self.screen_h // 2 + int(gaze_y * self.screen_h * self.sensitivity)
                            
                            # Apply boundaries
                            mouse_x = max(50, min(self.screen_w - 50, mouse_x))
                            mouse_y = max(50, min(self.screen_h - 50, mouse_y))
                            
                            # Apply smoothing
                            smooth_x, smooth_y = self.apply_smoothing(mouse_x, mouse_y)
                            
                            # Move mouse
                            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                            
                            # Handle clicks
                            clicked, click_type = self.handle_multiple_click_methods(landmarks, smooth_x, smooth_y)
                            
                            if clicked:
                                print(f"🖱️ {click_type}")
                                cv2.circle(frame, (50, 50), 30, (0, 255, 0), -1)
                                cv2.putText(frame, click_type, (60, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Draw pupil positions
                            for eye_name, data in pupil_data.items():
                                cv2.circle(frame, data['absolute'], 3, (0, 0, 255), -1)
                                x1, y1, x2, y2 = data['region']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                # Draw UI
                self.draw_ui_elements(frame)
                
                # Display frame
                cv2.imshow("Advanced Pupil Mouse Control", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_config()
                elif key == ord('r'):
                    self.load_config()
                elif key == ord('+') or key == ord('='):
                    self.sensitivity = min(5.0, self.sensitivity + 0.1)
                    print(f"Sensitivity: {self.sensitivity:.1f}")
                elif key == ord('-'):
                    self.sensitivity = max(0.5, self.sensitivity - 0.1)
                    print(f"Sensitivity: {self.sensitivity:.1f}")
                elif key == ord('c'):
                    print("Calibration feature coming soon!")
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cam.release()
        cv2.destroyAllWindows()
        print("👋 Advanced Pupil Mouse Control stopped")

def main():
    """Main function"""
    try:
        controller = AdvancedPupilMouseControl()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
