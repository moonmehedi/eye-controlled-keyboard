#!/usr/bin/env python3
"""
Quick test script to verify the ultra-advanced eye tracker works
"""

import sys
import os

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"   Version: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print("✅ SciPy imported successfully")
        print(f"   Version: {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import pyautogui
        print("✅ PyAutoGUI imported successfully")
        print(f"   Version: {pyautogui.__version__}")
    except ImportError as e:
        print(f"❌ PyAutoGUI import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera access."""
    print("\nTesting camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera test successful")
                print(f"   Frame shape: {frame.shape}")
            else:
                print("❌ Could not read from camera")
                return False
            cap.release()
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False
    
    return True

def test_mediapipe():
    """Test MediaPipe face detection."""
    print("\nTesting MediaPipe face detection...")
    try:
        import cv2
        import mediapipe as mp
        
        # Initialize face mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe FaceMesh initialized successfully")
        face_mesh.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ULTRA-ADVANCED EYE TRACKER - SYSTEM TEST")
    print("=" * 60)
    
    print(f"Python: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_camera():
        tests_passed += 1
    
    if test_mediapipe():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to run ultra_eye_tracker.py")
        print()
        print("To start the eye tracker:")
        print("  python ultra_eye_tracker.py")
        print()
        print("Controls:")
        print("  C - Calibrate")
        print("  Q - Quit")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print()
        print("Common solutions:")
        print("1. Make sure virtual environment is activated")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check camera permissions")

if __name__ == "__main__":
    main()





