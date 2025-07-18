#!/usr/bin/env python3
"""
Simple test script to verify pupil tracking functionality
"""
import cv2
import mediapipe as mp
import numpy as np
import time

def test_camera_and_detection():
    """Test basic camera and face detection"""
    print("🧪 Testing Camera and Face Detection")
    print("=" * 50)
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    # Initialize face mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    print("✅ Camera and MediaPipe initialized successfully")
    print("Look at the camera and press 'q' to quit")
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("❌ Error: Failed to read frame")
                break
            
            frame_count += 1
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Check for face landmarks
            if results.multi_face_landmarks:
                detection_count += 1
                
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Draw some key landmarks
                    h, w = frame.shape[:2]
                    
                    # Eye landmarks
                    eye_landmarks = [33, 133, 362, 263, 159, 145, 386, 374]
                    
                    for landmark_id in eye_landmarks:
                        x = int(landmarks[landmark_id].x * w)
                        y = int(landmarks[landmark_id].y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw eye regions
                    # Left eye
                    left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                    left_points = np.array([(int(p.x * w), int(p.y * h)) for p in left_eye])
                    cv2.drawContours(frame, [left_points], -1, (255, 0, 0), 1)
                    
                    # Right eye
                    right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
                    right_points = np.array([(int(p.x * w), int(p.y * h)) for p in right_eye])
                    cv2.drawContours(frame, [right_points], -1, (255, 0, 0), 1)
                    
                    # Calculate simple gaze direction
                    left_center = np.mean(left_points, axis=0)
                    right_center = np.mean(right_points, axis=0)
                    gaze_center = (left_center + right_center) / 2
                    
                    # Draw gaze point
                    cv2.circle(frame, (int(gaze_center[0]), int(gaze_center[1])), 5, (0, 255, 255), -1)
                    
                    # Display gaze coordinates
                    cv2.putText(frame, f"Gaze: ({int(gaze_center[0])}, {int(gaze_center[1])})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Calculate simple eye aspect ratio
                    left_ear = np.linalg.norm(left_points[1] - left_points[5]) / np.linalg.norm(left_points[0] - left_points[3])
                    right_ear = np.linalg.norm(right_points[1] - right_points[5]) / np.linalg.norm(right_points[0] - right_points[3])
                    avg_ear = (left_ear + right_ear) / 2
                    
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Status
                    if avg_ear < 0.22:
                        cv2.putText(frame, "BLINK DETECTED", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display stats
            if frame_count > 0:
                detection_rate = (detection_count / frame_count) * 100
                cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate FPS
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Pupil Tracking Test", frame)
            
            # Check for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Error during test: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n📊 Test Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Faces detected: {detection_count}")
        if frame_count > 0:
            print(f"Detection rate: {(detection_count/frame_count)*100:.1f}%")
        
        if detection_count > frame_count * 0.7:
            print("✅ Test PASSED - Good detection rate")
            return True
        else:
            print("⚠️ Test WARNING - Low detection rate")
            return False

def main():
    """Main test function"""
    print("🎯 Pupil Mouse Control - System Test")
    print("This will test your camera and face detection")
    print("Make sure you have good lighting and look at the camera")
    print()
    
    # Test camera and detection
    success = test_camera_and_detection()
    
    if success:
        print("\n✅ All tests passed! Your system is ready for pupil mouse control.")
        print("You can now run one of the main scripts:")
        print("- python pupil_mouse_control.py")
        print("- python advanced_pupil_mouse.py")
    else:
        print("\n⚠️ Some tests failed. Please check:")
        print("- Camera is working properly")
        print("- You have good lighting")
        print("- Face is clearly visible")
        print("- Required packages are installed")

if __name__ == "__main__":
    main()
