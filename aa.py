import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize components 
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Add safety and performance variables
last_click_time = 0
click_cooldown = 1.0  # seconds between clicks
pyautogui.FAILSAFE = True  # Enable failsafe

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read from camera")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        
        if landmark_points:
            landmarks = landmark_points[0].landmark
            
            # Draw eye tracking points
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)
            
            # Blink detection for clicking
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            
            # Check for blink with cooldown
            if (left[0].y - left[1].y) < 0.004:
                current_time = time.time()
                if current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time
                    print("Click detected!")
        
        # Display frame
        cv2.imshow('Eye Controlled Mouse', frame)
        
        # Exit on 'q' key or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up resources
    cam.release()
    cv2.destroyAllWindows()
    print("Camera and windows closed successfully")
