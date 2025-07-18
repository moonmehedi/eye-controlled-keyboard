# 👁️ Pupil Mouse Control for Paralyzed Users

A comprehensive eye-tracking mouse control system designed specifically for completely paralyzed individuals. This system uses advanced computer vision to track pupil movement and provides multiple clicking methods for accessibility.

## 🎯 Features

### Core Functionality
- **High-precision pupil tracking** - Tracks pupil position for accurate mouse control
- **Multiple click methods** - Blink, dwell, long blink, and wink options
- **Smooth movement** - Advanced smoothing algorithms to reduce jitter
- **Customizable sensitivity** - Adjustable to individual needs
- **Safety boundaries** - Prevents accidental clicks outside safe zones

### Accessibility Features
- **Dwell clicking** - Click by holding gaze steady (no physical movement required)
- **Multiple blink patterns** - Different blink types for different actions
- **Configuration saving** - Save personalized settings
- **Performance monitoring** - Real-time FPS and detection rate display
- **Zone-based navigation** - Simplified screen navigation

## 📋 Requirements

### Hardware
- **Camera**: Any USB webcam (720p or higher recommended)
- **Lighting**: Good, even lighting on the face
- **Positioning**: Camera should be at eye level, 50-80cm away

### Software
- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- PyAutoGUI (`pyautogui`)
- NumPy (`numpy`)

## 🛠️ Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Test your system**:
   ```bash
   python test_pupil_system.py
   ```

## 🚀 Usage

### Basic Pupil Mouse Control
```bash
python pupil_mouse_control.py
```

### Advanced Version (Recommended)
```bash
python advanced_pupil_mouse.py
```

### System Test
```bash
python test_pupil_system.py
```

## 🎮 Controls

### Mouse Movement
- **Look around** - Move your eyes to control the mouse cursor
- **Sensitivity** - Adjustable with +/- keys during runtime

### Clicking Methods

#### 1. Blink Clicking
- **Quick blink** (0.1-0.3 seconds) - Left click
- **Long blink** (0.5+ seconds) - Right click

#### 2. Dwell Clicking
- **Hold gaze steady** for 1.5 seconds - Left click
- **Movement threshold** - 25 pixels (configurable)

#### 3. Wink Clicking (Optional)
- **Left wink** - Left click
- **Right wink** - Right click

### Keyboard Controls
- **'q'** - Quit the application
- **'s'** - Save current settings
- **'r'** - Reset to defaults
- **'c'** - Start calibration (coming soon)
- **'+'** - Increase sensitivity
- **'-'** - Decrease sensitivity
- **'d'** - Toggle dwell clicking

## ⚙️ Configuration

### Automatic Configuration
The system automatically saves your preferences in `pupil_mouse_config.json`:

```json
{
  "sensitivity": 1.8,
  "smoothing_factor": 10,
  "dwell_time": 1.5,
  "click_cooldown": 0.6,
  "ear_threshold": 0.22
}
```

### Manual Adjustments
You can edit the configuration file directly or use the runtime controls.

## 📊 Performance Optimization

### For Best Results:
1. **Lighting**: Use even, soft lighting on your face
2. **Camera position**: Eye level, 50-80cm distance
3. **Background**: Plain, non-distracting background
4. **Stability**: Keep head as still as possible
5. **Calibration**: Run the test script first to verify detection

### Troubleshooting:
- **Low detection rate**: Improve lighting, adjust camera angle
- **Jittery movement**: Increase smoothing factor
- **Slow response**: Decrease smoothing factor
- **Accidental clicks**: Increase click cooldown time

## 🔧 Customization

### Sensitivity Adjustment
```python
# In the code, adjust these values:
self.sensitivity = 1.8  # Higher = more sensitive
self.smoothing_factor = 10  # Higher = smoother but slower
```

### Click Thresholds
```python
# Blink detection
self.ear_threshold = 0.22  # Lower = more sensitive to blinks
self.min_blink_frames = 2  # Minimum frames for blink
self.max_blink_frames = 8  # Maximum frames for blink
```

### Dwell Settings
```python
# Dwell clicking
self.dwell_time = 1.5  # Seconds to hold gaze
self.dwell_threshold = 25  # Pixels movement tolerance
```

## 🎯 Scripts Overview

### 1. `pupil_mouse_control.py`
- Basic pupil tracking with blink and dwell clicking
- Good for initial testing and simple usage

### 2. `advanced_pupil_mouse.py`
- Full-featured version with all accessibility options
- Configuration saving/loading
- Performance monitoring
- Multiple click methods
- Recommended for regular use

### 3. `test_pupil_system.py`
- System verification and troubleshooting
- Performance testing
- Detection rate monitoring
- Run this first to verify your setup

### 4. `eye_mouse_control.py`
- Alternative implementation using eye movement
- Less precise but simpler
- Good backup option

## 🛡️ Safety Features

### Built-in Safety
- **Fail-safe disabled** - Won't stop if mouse moves to corner (important for paralyzed users)
- **Click cooldown** - Prevents accidental rapid clicking
- **Boundary limits** - Keeps mouse within safe screen areas
- **Smooth movement** - Reduces erratic cursor behavior

### Emergency Controls
- **'q' key** - Always available to quit
- **Camera disconnection** - Automatically stops the system
- **Exception handling** - Graceful error recovery

## 🔍 Technical Details

### Eye Tracking Method
1. **Face detection** using MediaPipe Face Mesh
2. **Eye region extraction** from facial landmarks
3. **Pupil detection** using adaptive thresholding
4. **Gaze calculation** from pupil position relative to eye center
5. **Coordinate mapping** to screen space with smoothing

### Blink Detection
- **Eye Aspect Ratio (EAR)** calculation
- **Temporal filtering** to distinguish blinks from normal eye movement
- **Dual-threshold system** for different blink types

### Performance Optimizations
- **Efficient processing** pipeline
- **Memory management** for real-time operation
- **Adaptive quality** based on system performance

## 📞 Support

### Common Issues:
1. **"Import cv2 could not be resolved"** - Install OpenCV: `pip install opencv-python`
2. **"Import mediapipe could not be resolved"** - Install MediaPipe: `pip install mediapipe`
3. **Camera not working** - Check camera permissions and connections
4. **Poor detection** - Improve lighting and camera positioning

### Getting Help:
- Check the test script output for diagnostic information
- Verify all dependencies are installed correctly
- Ensure camera is working with other applications first

## 📈 Future Enhancements

### Planned Features:
- **Voice commands** - Additional control method
- **Gesture recognition** - Facial expression commands
- **Multiple monitor support** - Extended desktop control
- **Mobile app** - Remote configuration and monitoring
- **Machine learning** - Adaptive learning for individual users

### Contribute:
This project is designed to help paralyzed individuals gain computer access. Contributions and improvements are welcome!

## 📄 License

This project is open-source and designed to help people with disabilities. Feel free to use, modify, and distribute as needed.

## 🙏 Acknowledgments

- **MediaPipe** team for facial landmark detection
- **OpenCV** community for computer vision tools
- **PyAutoGUI** for mouse control functionality
- **Accessibility community** for inspiration and requirements

---

*This system is designed to provide computer access for paralyzed individuals. While it can be used by anyone, the focus is on accessibility and ease of use for those with limited mobility.*
