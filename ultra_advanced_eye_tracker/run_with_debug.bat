@echo off
echo Ultra-Advanced Eye Tracker - Debug Run
echo ========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call "D:\eye_control\eye-controlled-keyboard\venv\Scripts\activate.bat"

echo.
echo Current Python:
where python
echo.

REM Run system test first
echo Running system test...
python test_system.py

echo.
echo ========================================
echo.
echo If all tests passed, the eye tracker will start in 5 seconds...
echo Press Ctrl+C to cancel
timeout /t 5

echo.
echo Starting Ultra-Advanced Eye Tracker...
echo Remember: Press 'C' to calibrate when the camera window opens
echo.

python ultra_eye_tracker.py

pause
