import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
from collections import deque

class EyeKeyboard: 
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Improved blink detection parameters
        self.blink_threshold = 0.15  # Start with higher threshold
        self.last_click_time = 0
        self.click_cooldown = 0.5  # Faster response
        self.running = True
        
        # Advanced blink detection
        self.ear_history = deque(maxlen=30)  # Longer history for stability
        self.baseline_ear = 0.25  # Will be calibrated
        self.blink_detected = False
        self.blink_frames = 0
        self.min_blink_frames = 3  # Minimum frames for valid blink
        self.calibration_frames = 0
        self.calibration_complete = False
        
        # Eye tracking for letter detection
        self.gaze_history = deque(maxlen=10)  # Smooth gaze tracking
        self.dwell_time = {}  # Time spent looking at each key
        self.dwell_threshold = 0.5  # seconds to dwell before highlighting
        
        # Keyboard layout
        self.keyboard_layout = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'BACKSPACE', 'ENTER', 'CLEAR']
        ]
        
        # Text management
        self.typed_text = ""
        self.text_queue = queue.Queue()
        
        # Button positions and tracking
        self.button_positions = {}
        self.current_key = None
        self.gaze_x = 0
        self.gaze_y = 0
        
        # Debug info
        self.debug_info = {
            'ear': 0,
            'baseline': 0,
            'threshold': 0,
            'gaze_key': None,
            'blink_frames': 0
        }
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        """Create the virtual keyboard GUI"""
        self.root = tk.Tk()
        self.root.title("Eye-Controlled Keyboard for Paralyzed Users")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(self.root, text="Eye-Controlled Keyboard", 
                              font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Text display area with scrolling
        text_frame = tk.Frame(self.root, bg='#f0f0f0')
        text_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        tk.Label(text_frame, text="Typed Text:", font=("Arial", 12, "bold"), 
                bg='#f0f0f0').pack(anchor='w')
        
        self.text_display = scrolledtext.ScrolledText(
            text_frame, 
            height=8, 
            width=80, 
            font=("Arial", 14),
            bg='white',
            fg='black',
            wrap=tk.WORD
        )
        self.text_display.pack(fill='both', expand=True, pady=5)
        
        # Keyboard frame
        keyboard_frame = tk.Frame(self.root, bg='#f0f0f0')
        keyboard_frame.pack(pady=10)
        
        # Create keyboard buttons with click handlers
        self.buttons = {}
        for row_idx, row in enumerate(self.keyboard_layout):
            row_frame = tk.Frame(keyboard_frame, bg='#f0f0f0')
            row_frame.pack(pady=3)
            
            for col_idx, key in enumerate(row):
                # Button styling based on key type
                if key == 'SPACE':
                    btn = tk.Button(row_frame, text=key, width=25, height=2, 
                                  font=("Arial", 10, "bold"), bg='#87CEEB', 
                                  activebackground='#4682B4', relief='raised')
                elif key in ['BACKSPACE', 'ENTER', 'CLEAR']:
                    btn = tk.Button(row_frame, text=key, width=10, height=2, 
                                  font=("Arial", 10, "bold"), bg='#FFB6C1', 
                                  activebackground='#FF69B4', relief='raised')
                elif key.isdigit():
                    btn = tk.Button(row_frame, text=key, width=6, height=2, 
                                  font=("Arial", 10, "bold"), bg='#DDA0DD', 
                                  activebackground='#9370DB', relief='raised')
                else:
                    btn = tk.Button(row_frame, text=key, width=6, height=2, 
                                  font=("Arial", 10, "bold"), bg='#98FB98', 
                                  activebackground='#32CD32', relief='raised')
                
                # Add click command for manual clicking
                btn.config(command=lambda k=key: self.manual_key_press(k))
                btn.pack(side=tk.LEFT, padx=2)
                self.buttons[key] = btn
        
        # Status and control frame
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10)
        
        # Status display
        self.status_label = tk.Label(control_frame, text="Status: Ready", 
                                   font=("Arial", 12), bg='#f0f0f0', fg='blue')
        self.status_label.pack(pady=5)
        
        # Gaze info
        self.gaze_label = tk.Label(control_frame, text="Gaze: Not tracking", 
                                 font=("Arial", 10), bg='#f0f0f0', fg='green')
        self.gaze_label.pack(pady=2)
        
        # Blink status
        self.blink_label = tk.Label(control_frame, text="Blink: Calibrating...", 
                                  font=("Arial", 10), bg='#f0f0f0', fg='red')
        self.blink_label.pack(pady=2)
        
        # Calibration status
        self.calibration_label = tk.Label(control_frame, text="Calibration: Look normal for 3 seconds", 
                                        font=("Arial", 10), bg='#f0f0f0', fg='purple')
        self.calibration_label.pack(pady=2)
        
        # Debug info display
        self.debug_label = tk.Label(control_frame, text="Debug: Ready", 
                                  font=("Arial", 9), bg='#f0f0f0', fg='gray')
        self.debug_label.pack(pady=2)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.start_btn = tk.Button(button_frame, text="🎯 Start Eye Tracking", 
                                 command=self.start_eye_tracking, 
                                 bg='#32CD32', fg='white', font=("Arial", 12, "bold"))
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop", 
                                command=self.stop_tracking, 
                                bg='#FF4500', fg='white', font=("Arial", 12, "bold"))
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(button_frame, text="🗑 Clear Text", 
                                 command=self.clear_text, 
                                 bg='#FFD700', fg='black', font=("Arial", 12, "bold"))
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Calibration button
        self.calibrate_btn = tk.Button(button_frame, text="🎯 Calibrate Blink", 
                                     command=self.start_calibration, 
                                     bg='#9370DB', fg='white', font=("Arial", 10, "bold"))
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        # Store button positions after GUI is created
        self.root.after(500, self.update_button_positions)
        
        # Start text update thread
        self.start_text_updater()
        
    def start_calibration(self):
        """Start blink calibration"""
        self.calibration_frames = 0
        self.calibration_complete = False
        self.ear_history.clear()
        self.calibration_label.config(text="Calibrating: Look normal, don't blink for 3 seconds...")
        print("Starting calibration - keep eyes open normally for 3 seconds")
        
    def update_button_positions(self):
        """Update button positions for gaze mapping"""
        for key, button in self.buttons.items():
            try:
                button.update_idletasks()
                x = button.winfo_rootx()
                y = button.winfo_rooty()
                width = button.winfo_width()
                height = button.winfo_height()
                
                self.button_positions[key] = {
                    'x': x, 'y': y, 'width': width, 'height': height,
                    'center_x': x + width // 2, 'center_y': y + height // 2
                }
            except Exception as e:
                print(f"Error updating button position for {key}: {e}")
    
    def manual_key_press(self, key):
        """Handle manual button clicks"""
        print(f"Manual key press: {key}")
        self.type_key(key)
        self.flash_button(key)
        
    def start_text_updater(self):
        """Start the text update thread"""
        def update_text():
            while True:
                try:
                    text_update = self.text_queue.get(timeout=0.1)
                    if text_update == "STOP":
                        break
                    self.text_display.delete(1.0, tk.END)
                    self.text_display.insert(1.0, text_update)
                    self.text_display.see(tk.END)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Text update error: {e}")
        
        self.text_thread = threading.Thread(target=update_text, daemon=True)
        self.text_thread.start()
        
    def start_eye_tracking(self):
        """Start eye tracking"""
        if not self.running:
            self.running = True
            self.tracking_thread = threading.Thread(target=self.eye_tracking_loop, daemon=True)
            self.tracking_thread.start()
            self.status_label.config(text="Status: Eye tracking active", fg='green')
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
        
    def stop_tracking(self):
        """Stop eye tracking"""
        self.running = False
        self.status_label.config(text="Status: Stopped", fg='red')
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
    def clear_text(self):
        """Clear the text display"""
        self.typed_text = ""
        self.text_queue.put("")
        
    def get_key_at_gaze(self, screen_x, screen_y):
        """Get the key at the current gaze position with improved accuracy"""
        best_key = None
        min_distance = float('inf')
        
        for key, pos in self.button_positions.items():
            # Check if point is inside button bounds
            if (pos['x'] <= screen_x <= pos['x'] + pos['width'] and
                pos['y'] <= screen_y <= pos['y'] + pos['height']):
                return key
            
            # Calculate distance to button center for fallback
            dx = screen_x - pos['center_x']
            dy = screen_y - pos['center_y']
            distance = (dx * dx + dy * dy) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_key = key
        
        # Return closest key if within reasonable range
        if min_distance < 150:  # Increased tolerance
            return best_key
        return None
    
    def update_gaze_tracking(self, key):
        """Update gaze tracking with dwell time"""
        current_time = time.time()
        
        if key:
            if key not in self.dwell_time:
                self.dwell_time[key] = current_time
            elif current_time - self.dwell_time[key] > self.dwell_threshold:
                # Key has been looked at long enough
                self.current_key = key
                return True
        else:
            # Clear dwell times when not looking at any key
            self.dwell_time.clear()
            
        return False
        
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio with much more accurate landmarks"""
        try:
            # Use the most accurate eye landmarks for EAR calculation
            # Left eye (6 points)
            left_eye = [
                landmarks[33],   # Left corner
                landmarks[160],  # Top 1
                landmarks[158],  # Top 2
                landmarks[133],  # Right corner
                landmarks[153],  # Bottom 2
                landmarks[144]   # Bottom 1
            ]
            
            # Right eye (6 points)
            right_eye = [
                landmarks[362],  # Left corner
                landmarks[385],  # Top 1
                landmarks[387],  # Top 2
                landmarks[263],  # Right corner
                landmarks[373],  # Bottom 2
                landmarks[380]   # Bottom 1
            ]
            
            def compute_ear(eye_landmarks):
                # Calculate the euclidean distances between the vertical eye landmarks
                A = np.sqrt((eye_landmarks[1].x - eye_landmarks[5].x)**2 + 
                           (eye_landmarks[1].y - eye_landmarks[5].y)**2)
                B = np.sqrt((eye_landmarks[2].x - eye_landmarks[4].x)**2 + 
                           (eye_landmarks[2].y - eye_landmarks[4].y)**2)
                
                # Calculate the euclidean distance between the horizontal eye landmarks
                C = np.sqrt((eye_landmarks[0].x - eye_landmarks[3].x)**2 + 
                           (eye_landmarks[0].y - eye_landmarks[3].y)**2)
                
                # Calculate the eye aspect ratio
                ear = (A + B) / (2.0 * C)
                return ear
            
            # Calculate EAR for both eyes
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            
            # Return the average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            return avg_ear
            
        except Exception as e:
            print(f"EAR calculation error: {e}")
            return 0.3  # Safe default value
    
    def detect_blink_advanced(self, ear):
        """Advanced blink detection with multiple validation layers"""
        current_time = time.time()
        
        # Stage 1: Check if EAR is below threshold
        if ear < self.blink_threshold:
            self.blink_frames += 1
            
            # Stage 2: Validate blink duration (must be between 3-15 frames)
            if self.blink_frames >= 3 and self.blink_frames <= 15:
                if not self.blink_detected:
                    self.blink_detected = True
                    return True
        else:
            # Eyes are open - reset blink detection
            if self.blink_detected:
                self.blink_detected = False
                
            # Reset frame counter only if eyes have been open for a bit
            if self.blink_frames > 0:
                self.blink_frames = max(0, self.blink_frames - 1)
                
        return False
        
    def type_key(self, key):
        """Handle typing a key"""
        if key == 'SPACE':
            self.typed_text += ' '
        elif key == 'BACKSPACE':
            self.typed_text = self.typed_text[:-1]
        elif key == 'ENTER':
            self.typed_text += '\n'
        elif key == 'CLEAR':
            self.typed_text = ""
        else:
            self.typed_text += key
            
        # Update text display
        self.text_queue.put(self.typed_text)
        print(f"Typed: {key} | Text: '{self.typed_text}'")
        
    def flash_button(self, key):
        """Flash button when pressed"""
        if key in self.buttons:
            original_color = self.buttons[key].cget('bg')
            self.buttons[key].config(bg='yellow')
            self.root.after(300, lambda: self.buttons[key].config(bg=original_color))
            
    def highlight_key(self, key):
        """Highlight the key being looked at"""
        # Reset all buttons
        for k, btn in self.buttons.items():
            if k == 'SPACE':
                btn.config(bg='#87CEEB')
            elif k in ['BACKSPACE', 'ENTER', 'CLEAR']:
                btn.config(bg='#FFB6C1')
            elif k.isdigit():
                btn.config(bg='#DDA0DD')
            else:
                btn.config(bg='#98FB98')
        
        # Highlight current key
        if key and key in self.buttons:
            self.buttons[key].config(bg='#FFFF00')  # Bright yellow
            
    def eye_tracking_loop(self):
        """Main eye tracking loop with improved blink detection"""
        print("Starting eye tracking loop...")
        
        while self.running:
            ret, frame = self.cam.read()
            if not ret:
                print("Failed to read camera frame")
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            frame_h, frame_w = frame.shape[:2]
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Calculate gaze position using multiple eye landmarks for stability
                    left_eye_landmarks = [landmarks[33], landmarks[133], landmarks[157], landmarks[158], landmarks[159], landmarks[160]]
                    right_eye_landmarks = [landmarks[362], landmarks[263], landmarks[374], landmarks[373], landmarks[380], landmarks[385]]
                    
                    # Average eye positions
                    left_eye_center = np.mean([(lm.x, lm.y) for lm in left_eye_landmarks], axis=0)
                    right_eye_center = np.mean([(lm.x, lm.y) for lm in right_eye_landmarks], axis=0)
                    
                    # Calculate overall gaze position
                    gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
                    
                    # Convert to screen coordinates
                    screen_x = int(gaze_x * self.screen_w)
                    screen_y = int(gaze_y * self.screen_h)
                    
                    # Smooth gaze tracking
                    self.gaze_history.append((screen_x, screen_y))
                    if len(self.gaze_history) > 1:
                        avg_x = int(np.mean([pos[0] for pos in self.gaze_history]))
                        avg_y = int(np.mean([pos[1] for pos in self.gaze_history]))
                        screen_x, screen_y = avg_x, avg_y
                    
                    # Get key at gaze position
                    gaze_key = self.get_key_at_gaze(screen_x, screen_y)
                    
                    # Update gaze tracking with dwell time
                    key_ready = self.update_gaze_tracking(gaze_key)
                    
                    # Update UI with gaze information
                    self.root.after(1, lambda: self.gaze_label.config(
                        text=f"Gaze: ({screen_x}, {screen_y}) -> {gaze_key or 'None'} {'✓' if key_ready else ''}"))
                    
                    if gaze_key:
                        self.root.after(1, lambda k=gaze_key: self.highlight_key(k))
                    else:
                        self.root.after(1, lambda: self.highlight_key(None))
                    
                    # Calculate EAR for blink detection
                    ear = self.calculate_ear(landmarks)
                    self.ear_history.append(ear)
                    
                    # Auto-calibration during first frames
                    if not self.calibration_complete:
                        self.calibration_frames += 1
                        if self.calibration_frames < 150:  # 5 seconds at 30 FPS
                            self.root.after(1, lambda: self.calibration_label.config(
                                text=f"Calibrating... {150 - self.calibration_frames} frames left"))
                        else:
                            if len(self.ear_history) > 100:
                                # Calculate baseline from stable readings
                                stable_ears = [e for e in self.ear_history if 0.2 < e < 0.5]
                                if len(stable_ears) > 50:
                                    self.baseline_ear = np.mean(stable_ears)
                                    self.blink_threshold = self.baseline_ear * 0.75  # 75% of baseline
                                    self.calibration_complete = True
                                    self.root.after(1, lambda: self.calibration_label.config(
                                        text=f"✅ Calibrated! Baseline: {self.baseline_ear:.3f}"))
                                    print(f"✅ Calibration complete! Baseline: {self.baseline_ear:.3f}, Threshold: {self.blink_threshold:.3f}")
                                else:
                                    # Reset calibration if not enough stable readings
                                    self.calibration_frames = 0
                                    self.ear_history.clear()
                                    self.root.after(1, lambda: self.calibration_label.config(
                                        text="❌ Calibration failed, restarting..."))
                    
                    # Blink detection - only if calibrated
                    if self.calibration_complete:
                        # Use advanced blink detection
                        blink_occurred = self.detect_blink_advanced(ear)
                        
                        # Update debug info
                        self.debug_info.update({
                            'ear': ear,
                            'baseline': self.baseline_ear,
                            'threshold': self.blink_threshold,
                            'gaze_key': gaze_key,
                            'blink_frames': self.blink_frames
                        })
                        
                        # Update debug display
                        self.root.after(1, lambda: self.debug_label.config(
                            text=f"EAR: {ear:.3f} | Threshold: {self.blink_threshold:.3f} | Frames: {self.blink_frames}"))
                        
                        # Handle blink action
                        if blink_occurred:
                            current_time = time.time()
                            if current_time - self.last_click_time > self.click_cooldown:
                                if self.current_key:
                                    print(f"🎯 LEGITIMATE BLINK DETECTED! Typing: {self.current_key}")
                                    self.root.after(1, lambda: self.type_key(self.current_key))
                                    self.root.after(1, lambda: self.flash_button(self.current_key))
                                    self.last_click_time = current_time
                                    
                                    self.root.after(1, lambda: self.blink_label.config(
                                        text=f"✅ TYPED '{self.current_key}'!", fg='green'))
                                else:
                                    self.root.after(1, lambda: self.blink_label.config(
                                        text="❌ No key selected", fg='red'))
                        else:
                            # Update blink status based on current state
                            if ear < self.blink_threshold:
                                self.root.after(1, lambda: self.blink_label.config(
                                    text=f"👁️ Blinking... ({self.blink_frames} frames)", fg='orange'))
                            else:
                                self.root.after(1, lambda: self.blink_label.config(
                                    text=f"👀 Ready to blink", fg='blue'))
                    
                    # Draw enhanced eye tracking visualization
                    # Draw gaze position
                    gaze_screen_x = int(gaze_x * frame_w)
                    gaze_screen_y = int(gaze_y * frame_h)
                    cv2.circle(frame, (gaze_screen_x, gaze_screen_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"Gaze: {gaze_key or 'None'}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw EAR info
                    if self.calibration_complete:
                        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Threshold: {self.blink_threshold:.3f}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Blink Frames: {self.blink_frames}", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Show blink status
                        if ear < self.blink_threshold:
                            cv2.putText(frame, "BLINKING!", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "CALIBRATING...", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Draw eye landmarks
                    for landmark in left_eye_landmarks + right_eye_landmarks:
                        x = int(landmark.x * frame_w)
                        y = int(landmark.y * frame_h)
                        cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
                        
            # Display frame
            cv2.imshow('Eye Tracking - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        print("Eye tracking loop ended")
        self.cam.release()
        cv2.destroyAllWindows()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        self.text_queue.put("STOP")
        if hasattr(self, 'cam'):
            self.cam.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    print("Starting Eye-Controlled Keyboard...")
    app = EyeKeyboard()
    app.run()
        
    def create_gui(self):
        """Create the virtual keyboard GUI"""
        self.root = tk.Tk()
        self.root.title("Eye-Controlled Keyboard")
        self.root.geometry("800x600")
        
        # Text display area
        self.text_display = tk.Text(self.root, height=10, width=80, font=("Arial", 12))
        self.text_display.pack(pady=10)
        
        # Keyboard frame
        keyboard_frame = tk.Frame(self.root)
        keyboard_frame.pack(pady=10)
        
        # Create keyboard buttons
        self.buttons = {}
        for row_idx, row in enumerate(self.keyboard_layout):
            row_frame = tk.Frame(keyboard_frame)
            row_frame.pack(pady=2)
            
            for col_idx, key in enumerate(row):
                if key == 'SPACE':
                    btn = tk.Button(row_frame, text=key, width=20, height=2, 
                                  font=("Arial", 10), bg='lightblue')
                elif key in ['BACKSPACE', 'ENTER']:
                    btn = tk.Button(row_frame, text=key, width=12, height=2, 
                                  font=("Arial", 10), bg='lightcoral')
                else:
                    btn = tk.Button(row_frame, text=key, width=5, height=2, 
                                  font=("Arial", 10), bg='lightgreen')
                
                btn.pack(side=tk.LEFT, padx=2)
                self.buttons[key] = btn
                
                # Store button position for gaze mapping
                self.root.after(100, lambda k=key, b=btn: self.store_button_position(k, b))
        
        # Status label
        self.status_label = tk.Label(self.root, text="Status: Ready", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        start_btn = tk.Button(control_frame, text="Start Eye Tracking", 
                             command=self.start_eye_tracking, bg='green', fg='white')
        start_btn.pack(side=tk.LEFT, padx=5)
        
        stop_btn = tk.Button(control_frame, text="Stop", 
                            command=self.stop_tracking, bg='red', fg='white')
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(control_frame, text="Clear Text", 
                             command=self.clear_text, bg='orange', fg='white')
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Add gaze indicator
        self.gaze_indicator = tk.Label(self.root, text="●", font=("Arial", 20), fg='red')
        self.gaze_indicator.place(x=0, y=0)
        
    def store_button_position(self, key, button):
        """Store button positions for gaze mapping"""
        try:
            button.update_idletasks()
            x = button.winfo_rootx()
            y = button.winfo_rooty()
            width = button.winfo_width()
            height = button.winfo_height()
            
            self.button_positions[key] = {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'center_x': x + width // 2,
                'center_y': y + height // 2
            }
        except:
            pass
        
    def start_eye_tracking(self):
        """Start eye tracking in a separate thread"""
        self.running = True
        self.tracking_thread = threading.Thread(target=self.eye_tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        self.status_label.config(text="Status: Eye tracking active")
        
    def stop_tracking(self):
        """Stop eye tracking"""
        self.running = False
        self.status_label.config(text="Status: Stopped")
        
    def clear_text(self):
        """Clear the text display"""
        self.text_display.delete(1.0, tk.END)
        self.typed_text = ""
        
    def get_key_at_position(self, screen_x, screen_y):
        """Determine which key is being looked at based on screen position"""
        # Convert screen coordinates to window coordinates
        window_x = screen_x - self.root.winfo_rootx()
        window_y = screen_y - self.root.winfo_rooty()
        
        closest_key = None
        min_distance = float('inf')
        
        for key, pos in self.button_positions.items():
            # Check if gaze is within button bounds
            if (pos['x'] <= screen_x <= pos['x'] + pos['width'] and
                pos['y'] <= screen_y <= pos['y'] + pos['height']):
                return key
            
            # Find closest button as fallback
            distance = ((pos['center_x'] - screen_x) ** 2 + (pos['center_y'] - screen_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_key = key
        
        # Return closest key if within reasonable distance
        if min_distance < 100:  # pixels
            return closest_key
        return None
        
    def type_key(self, key):
        """Handle key typing"""
        if key == 'SPACE':
            self.typed_text += ' '
        elif key == 'BACKSPACE':
            self.typed_text = self.typed_text[:-1]
        elif key == 'ENTER':
            self.typed_text += '\n'
        else:
            self.typed_text += key
            
        # Update display
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.typed_text)
        
        # Highlight the pressed key
        if key in self.buttons:
            self.buttons[key].config(bg='yellow')
            self.root.after(500, lambda: self.reset_button_color(key))
            
    def reset_button_color(self, key):
        """Reset button color after highlighting"""
        if key in self.buttons:
            if key == 'SPACE':
                self.buttons[key].config(bg='lightblue')
            elif key in ['BACKSPACE', 'ENTER']:
                self.buttons[key].config(bg='lightcoral')
            else:
                self.buttons[key].config(bg='lightgreen')
                
    def highlight_current_key(self, key):
        """Highlight the key currently being looked at"""
        # Reset all buttons to default color
        for k, btn in self.buttons.items():
            if k == 'SPACE':
                btn.config(bg='lightblue')
            elif k in ['BACKSPACE', 'ENTER']:
                btn.config(bg='lightcoral')
            else:
                btn.config(bg='lightgreen')
        
        # Highlight current key
        if key and key in self.buttons:
            self.buttons[key].config(bg='lightyellow')
            
    def eye_tracking_loop(self):
        """Main eye tracking loop"""
        while self.running:
            ret, frame = self.cam.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks
            frame_h, frame_w, _ = frame.shape
            
            if landmark_points:
                landmarks = landmark_points[0].landmark
                
                # Get eye center for gaze tracking
                right_eye_center = landmarks[468]  # Right eye center
                left_eye_center = landmarks[473]   # Left eye center
                
                # Average both eyes for more stable tracking
                avg_x = (right_eye_center.x + left_eye_center.x) / 2
                avg_y = (right_eye_center.y + left_eye_center.y) / 2
                
                # Convert to screen coordinates
                screen_x = int(avg_x * self.screen_w)
                screen_y = int(avg_y * self.screen_h)
                
                # Update gaze position
                self.gaze_x = screen_x
                self.gaze_y = screen_y
                
                # Update gaze indicator position
                try:
                    window_x = screen_x - self.root.winfo_rootx()
                    window_y = screen_y - self.root.winfo_rooty()
                    self.gaze_indicator.place(x=max(0, min(window_x, self.root.winfo_width()-20)), 
                                            y=max(0, min(window_y, self.root.winfo_height()-20)))
                except:
                    pass
                
                # Draw eye tracking points on camera feed
                for id, landmark in enumerate(landmarks[468:478]):  # Eye landmarks
                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Get the key at current gaze position
                current_key = self.get_key_at_position(screen_x, screen_y)
                
                # Update status and highlight current key
                if current_key:
                    self.root.after(1, lambda: self.status_label.config(text=f"Status: Looking at {current_key}"))
                    self.root.after(1, lambda: self.highlight_current_key(current_key))
                    self.current_key = current_key
                else:
                    self.root.after(1, lambda: self.status_label.config(text="Status: Eye tracking active"))
                    self.root.after(1, lambda: self.highlight_current_key(None))
                
                # Blink detection for clicking
                left_eye_top = landmarks[159]
                left_eye_bottom = landmarks[145]
                right_eye_top = landmarks[386]
                right_eye_bottom = landmarks[374]
                
                # Calculate eye aspect ratio for both eyes
                left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
                right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
                avg_eye_height = (left_eye_height + right_eye_height) / 2
                
                # Draw blink detection points
                for landmark in [left_eye_top, left_eye_bottom, right_eye_top, right_eye_bottom]:
                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Check for blink (click) - adjusted threshold
                if avg_eye_height < 0.003:  # Stricter threshold
                    current_time = time.time()
                    if current_time - self.last_click_time > self.click_cooldown:
                        if self.current_key:
                            self.root.after(1, lambda: self.type_key(self.current_key))
                            print(f"Blink detected! Typing: {self.current_key}")
                            
                        self.last_click_time = current_time
                        
            # Show camera feed
            cv2.imshow('Eye Tracking Feed', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cam.release()
        cv2.destroyAllWindows()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if hasattr(self, 'cam'):
            self.cam.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    app = EyeKeyboard()
    app.run()
