import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

def main():
    """Simple eye tracking mouse control with blink to click"""
    
    # Initialize camera and face mesh
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    # Get screen dimensions
    screen_w, screen_h = pyautogui.size()
    
    # Safety settings
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    
    # Blink detection variables
    blink_threshold = 0.2
    blink_counter = 0
    last_click_time = 0
    click_cooldown = 0.8  # seconds between clicks
    
    print("🎯 Simple Eye Mouse Control")
    print("Instructions:")
    print("1. Look around to move the mouse")
    print("2. Blink to click")
    print("3. Press 'q' to quit")
    print("4. Move mouse to top-left corner for emergency stop")
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Get eye center positions for mouse control
                    # Left eye center
                    left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
                    left_eye_y = (landmarks[33].y + landmarks[133].y) / 2
                    
                    # Right eye center
                    right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
                    right_eye_y = (landmarks[362].y + landmarks[263].y) / 2
                    
                    # Average both eyes for gaze position
                    gaze_x = (left_eye_x + right_eye_x) / 2
                    gaze_y = (left_eye_y + right_eye_y) / 2
                    
                    # Map to screen coordinates (inverted X for natural movement)
                    mouse_x = int((1 - gaze_x) * screen_w)
                    mouse_y = int(gaze_y * screen_h)
                    
                    # Move mouse
                    try:
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0)
                    except pyautogui.FailSafeException:
                        print("Emergency stop activated!")
                        break
                    
                    # Simple blink detection using eye height
                    # Left eye height
                    left_eye_height = abs(landmarks[159].y - landmarks[145].y)
                    # Right eye height  
                    right_eye_height = abs(landmarks[386].y - landmarks[374].y)
                    
                    # Average eye height
                    avg_eye_height = (left_eye_height + right_eye_height) / 2
                    
                    # Blink detection
                    current_time = time.time()
                    
                    if avg_eye_height < blink_threshold:
                        blink_counter += 1
                    else:
                        if (blink_counter >= 2 and blink_counter <= 6 and 
                            current_time - last_click_time > click_cooldown):
                            
                            # Perform click
                            pyautogui.click()
                            last_click_time = current_time
                            print(f"🖱️ CLICK! (Eye height: {avg_eye_height:.3f})")
                            
                            # Visual feedback
                            cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
                        
                        blink_counter = 0
                    
                    # Display info
                    cv2.putText(frame, f"Eye Height: {avg_eye_height:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Blink Count: {blink_counter}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw eye landmarks for visual feedback
                    for landmark_id in [33, 133, 159, 145, 362, 263, 386, 374]:
                        x = int(landmarks[landmark_id].x * frame_w)
                        y = int(landmarks[landmark_id].y * frame_h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Display the frame
            cv2.imshow("Simple Eye Mouse Control", frame)
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("👋 Eye Mouse Control stopped")

if __name__ == "__main__":
    main()
