import cv2
import pyautogui
import numpy as np
from collections import deque
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import webbrowser
import urllib.parse
import subprocess
import os

# Try to import speech recognition for voice commands
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Google Speech Recognition not available. Install with: pip install SpeechRecognition")

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class SimpleEyeTracker:
    def __init__(self):
        # Load Haar Cascade classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Enhanced smoothing with weighted buffer
        self.smooth_buffer = deque(maxlen=15)  # Increased for much smoother tracking
        self.gaze_buffer = deque(maxlen=10)  # Increased for better stability
        
        # Blink detection variables (enhanced)
        self.last_blink_time = 0
        self.blink_cooldown = 0.4  # seconds (reduced for faster response)
        self.eyes_detected_history = deque(maxlen=8)  # Longer history for better detection
        self.blink_enabled = True  # Toggle blink-to-click
        self.blink_threshold = 2  # Frames with no eyes to consider as blink
        self.blink_detected_frame = 0
        
        # Control mode
        self.control_enabled = True
        
        # Sensitivity control (1.0 = normal, higher = more sensitive)
        self.sensitivity = 1.5
        self.speed_multiplier = 1.0
        self.fine_tune_mode = False  # Precision mode for small movements
        
        # Dead zone to reduce jitter (pixels)
        self.dead_zone = 15  # Increased for more stability
        self.adaptive_dead_zone = True  # Adjust dead zone based on movement
        self.last_cursor_pos = None
        
        # Cursor stabilization when hovering over buttons
        self.cursor_freeze_on_hover = True
        self.frozen_cursor_pos = None
        
        # Focus locking - hold gaze to lock cursor
        self.focus_lock_enabled = False
        self.focus_lock_threshold = 0.8  # seconds to lock
        self.focus_lock_start_time = None
        self.focus_locked = False
        self.locked_position = None
        
        # Advanced smoothing
        self.exponential_smoothing = True
        self.alpha = 0.25  # Exponential smoothing factor (0-1, lower = more smooth)
        
        # Enhanced calibration with 9 points for better accuracy
        self.calibration_mode = False
        self.interactive_calibration = True
        self.calibration_complete = False
        
        # Machine learning model for gaze mapping
        self.model_x_coeffs = None  # Polynomial coefficients for X-axis
        self.model_y_coeffs = None  # Polynomial coefficients for Y-axis
        self.model_accuracy = 0.0   # Model R² score
        
        self.calibration_points_targets = [
            (0.1, 0.1),   # Top-left
            (0.5, 0.1),   # Top-center
            (0.9, 0.1),   # Top-right
            (0.1, 0.5),   # Middle-left
            (0.5, 0.5),   # Center
            (0.9, 0.5),   # Middle-right
            (0.1, 0.9),   # Bottom-left
            (0.5, 0.9),   # Bottom-center
            (0.9, 0.9),   # Bottom-right
        ]
        self.calibration_validation_mode = False
        self.calibration_accuracy_score = 0
        self.current_calibration_point = 0
        self.calibration_samples = []  # All samples for model training
        self.current_point_samples = []  # Samples for current calibration point
        self.samples_per_point = 50  # Increased for better accuracy
        self.calibration_delay = 2.0  # Increased stabilization time
        self.calibration_start_time = None
        self.calibration_gaze_data = []  # Store gaze positions at calibration points
        
        self.calibration_data = {
            'min_x': float('inf'),
            'max_x': float('-inf'),
            'min_y': float('inf'),
            'max_y': float('-inf')
        }
        self.frames_processed = 0
        self.calibration_points = []
        
        # Tracking quality metrics
        self.tracking_confidence = 0.0
        self.last_valid_gaze = None
        
        # Detection improvement variables
        self.last_face_rect = None
        self.face_miss_counter = 0
        self.max_face_misses = 3
        
        # Volume control with keyboard simulation
        self.gesture_enabled = True
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            self.volume_range = self.volume.GetVolumeRange()
            self.min_volume = self.volume_range[0]
            self.max_volume = self.volume_range[1]
            self.volume_available = True
            print(f"✅ Volume control initialized successfully")
            print(f"   Volume range: {self.min_volume:.1f} to {self.max_volume:.1f}")
        except Exception as e:
            self.volume_available = False
            print(f"❌ Volume control NOT available: {e}")
            print("   Install with: pip install pycaw==20230407 comtypes")
        
        # Gesture state tracking
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.8
        self.gesture_threshold = 3  # frames to confirm gesture
        
        # Blink detection for clicking and volume control
        self.blink_enabled = True
        self.blink_threshold = 0.3
        self.last_blink_time = 0
        self.blink_cooldown = 0.25
        self.blink_history = deque(maxlen=8)
        self.eyes_closed_frames = 0
        self.eyes_open_frames = 0
        self.min_blink_frames = 2
        self.max_blink_frames = 8
        self.blink_detected_frame = 0
        
        # Enhanced blink detection state machine
        self.blink_state = "EYES_OPEN"  # States: EYES_OPEN, CLOSING, EYES_CLOSED, OPENING
        self.blink_confidence = 0.0
        self.stable_eye_count = 2  # Expected stable eye count
        self.blink_validation_frames = 3  # Frames to confirm state change
        
        # Blink pattern detection for volume control
        self.blink_pattern_buffer = []  # Stores timestamps of recent blinks
        self.pattern_window = 3.0  # seconds to detect pattern (increased for easier use)
        self.long_blink_threshold = 0.8  # seconds for long blink (mute)
        self.last_volume_action_time = 0
        self.volume_action_cooldown = 1.0  # Prevent rapid volume changes
        self.waiting_for_pattern = False  # Track if we're waiting to complete a pattern
        self.pattern_start_time = 0
        self.pattern_check_delay = 0.4  # Shorter delay to check for patterns
        self.volume_change_feedback = None  # For visual feedback
        self.volume_feedback_time = 0
        self.volume_feedback_duration = 2.0  # seconds to show feedback
        
        # One-eye-closed detection for volume down
        self.one_eye_history = deque(maxlen=15)  # Track single eye detections (increased buffer)
        self.one_eye_threshold = 8  # Frames with one eye to trigger volume down (increased for accuracy)
        self.last_one_eye_volume_time = 0
        self.one_eye_volume_cooldown = 1.5  # Cooldown between one-eye volume actions
        
        # Google Features Integration
        self.google_features_enabled = True
        self.voice_recognition_enabled = SPEECH_AVAILABLE
        self.last_google_action_time = 0
        self.google_action_cooldown = 2.0  # seconds between Google actions
        
        # Speech recognizer setup
        if SPEECH_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
        else:
            self.recognizer = None
            self.microphone = None
        
        # Google Chrome path detection
        self.chrome_path = self.find_chrome_path()
        
        # On-screen GUI buttons for eye control
        self.show_gui_buttons = True
        self.gui_buttons = []
        self.hovered_button = None
        self.button_hover_time = 0
        self.button_click_threshold = 0.8  # seconds to hover before auto-click (increased for stability)
        self.last_button_click_time = 0
        self.button_click_cooldown = 1.5
        self.initialize_gui_buttons()
    
    def initialize_gui_buttons(self):
        """Initialize on-screen clickable buttons"""
        # Button format: (x, y, width, height, label, action, color)
        btn_w, btn_h = 140, 40
        spacing = 10
        start_x = 10
        start_y = 220
        
        self.gui_buttons = [
            # Row 1 - Google Services
            {'x': start_x, 'y': start_y, 'w': btn_w, 'h': btn_h, 
             'label': 'Gmail', 'action': 'gmail', 'color': (0, 120, 255)},
            {'x': start_x + btn_w + spacing, 'y': start_y, 'w': btn_w, 'h': btn_h,
             'label': 'Drive', 'action': 'drive', 'color': (0, 200, 100)},
            {'x': start_x + (btn_w + spacing) * 2, 'y': start_y, 'w': btn_w, 'h': btn_h,
             'label': 'Docs', 'action': 'docs', 'color': (255, 180, 0)},
            
            # Row 2 - More Google Services
            {'x': start_x, 'y': start_y + btn_h + spacing, 'w': btn_w, 'h': btn_h,
             'label': 'YouTube', 'action': 'youtube', 'color': (0, 0, 255)},
            {'x': start_x + btn_w + spacing, 'y': start_y + btn_h + spacing, 'w': btn_w, 'h': btn_h,
             'label': 'Maps', 'action': 'maps', 'color': (100, 200, 100)},
            {'x': start_x + (btn_w + spacing) * 2, 'y': start_y + btn_h + spacing, 'w': btn_w, 'h': btn_h,
             'label': 'Calendar', 'action': 'calendar', 'color': (255, 100, 100)},
            
            # Row 3 - Voice Commands
            {'x': start_x, 'y': start_y + (btn_h + spacing) * 2, 'w': btn_w, 'h': btn_h,
             'label': 'Voice Search', 'action': 'voice_search', 'color': (180, 0, 180)},
            {'x': start_x + btn_w + spacing, 'y': start_y + (btn_h + spacing) * 2, 'w': btn_w, 'h': btn_h,
             'label': 'Translate', 'action': 'translate', 'color': (0, 180, 180)},
            {'x': start_x + (btn_w + spacing) * 2, 'y': start_y + (btn_h + spacing) * 2, 'w': btn_w, 'h': btn_h,
             'label': 'Chrome', 'action': 'chrome', 'color': (200, 200, 0)},
            
            # Row 4 - Controls
            {'x': start_x, 'y': start_y + (btn_h + spacing) * 3, 'w': btn_w, 'h': btn_h,
             'label': 'Calibrate', 'action': 'calibrate', 'color': (255, 128, 0)},
            {'x': start_x + btn_w + spacing, 'y': start_y + (btn_h + spacing) * 3, 'w': btn_w, 'h': btn_h,
             'label': 'Toggle GUI', 'action': 'toggle_gui', 'color': (128, 128, 128)},
        ]
    
    def check_button_hover(self, gaze_x, gaze_y):
        """Check if gaze is hovering over any button"""
        if not self.show_gui_buttons or gaze_x is None or gaze_y is None:
            return None
        
        for button in self.gui_buttons:
            if (button['x'] <= gaze_x <= button['x'] + button['w'] and
                button['y'] <= gaze_y <= button['y'] + button['h']):
                return button
        return None
    
    def execute_button_action(self, action):
        """Execute the action associated with a button"""
        current_time = time.time()
        
        # Prevent rapid clicks
        if current_time - self.last_button_click_time < self.button_click_cooldown:
            return
        
        self.last_button_click_time = current_time
        
        print(f"\n🎯 Button Clicked: {action}")
        
        # Google Services
        if action in ['gmail', 'drive', 'docs', 'sheets', 'calendar', 'maps', 'youtube', 'meet']:
            self.open_google_service(action)
        
        # Voice Commands
        elif action == 'voice_search':
            print("🔍 Voice Search - Speak your query...")
            command = self.listen_for_voice_command()
            if command:
                self.google_search(command)
        
        elif action == 'translate':
            print("🌍 Google Translate - Speak the text...")
            command = self.listen_for_voice_command()
            if command:
                self.google_translate(command)
        
        elif action == 'chrome':
            self.open_google_chrome()
        
        # Controls
        elif action == 'calibrate':
            self.start_calibration()
        
        elif action == 'toggle_gui':
            self.show_gui_buttons = not self.show_gui_buttons
            print(f"GUI Buttons: {'VISIBLE' if self.show_gui_buttons else 'HIDDEN'}")
    
    def draw_gui_buttons(self, frame):
        """Draw clickable buttons on the frame"""
        if not self.show_gui_buttons:
            return
        
        current_time = time.time()
        
        for button in self.gui_buttons:
            x, y, w, h = button['x'], button['y'], button['w'], button['h']
            color = button['color']
            label = button['label']
            
            # Check if this button is being hovered
            is_hovered = (self.hovered_button and 
                         self.hovered_button['label'] == button['label'])
            
            # Draw button background
            if is_hovered:
                # Highlight when hovered
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw progress bar for hover-to-click
                hover_duration = current_time - self.button_hover_time
                progress = min(hover_duration / self.button_click_threshold, 1.0)
                progress_w = int(w * progress)
                cv2.rectangle(frame, (x, y + h - 5), (x + progress_w, y + h), color, -1)
            else:
                # Normal button
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Draw button text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2 if is_hovered else 1
            text_color = (0, 0, 0) if is_hovered else (255, 255, 255)
            
            # Center text in button
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Draw title
        cv2.putText(frame, "Eye-Controlled Buttons", (10, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def preprocess_for_detection(self, gray):
        """Lightweight preprocessing for better detection"""
        # Simple histogram equalization only (fast)
        return cv2.equalizeHist(gray)
    
    def find_chrome_path(self):
        """Find Google Chrome installation path"""
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def open_google_chrome(self, url="https://www.google.com"):
        """Open Google Chrome browser"""
        try:
            if self.chrome_path:
                subprocess.Popen([self.chrome_path, url])
                print(f"🌐 Opened Chrome: {url}")
            else:
                webbrowser.open(url)
                print(f"🌐 Opened browser: {url}")
            return True
        except Exception as e:
            print(f"Error opening Chrome: {e}")
            return False
    
    def google_search(self, query):
        """Perform Google search"""
        try:
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            self.open_google_chrome(search_url)
            print(f"🔍 Google Search: {query}")
            return True
        except Exception as e:
            print(f"Error performing Google search: {e}")
            return False
    
    def google_translate(self, text, target_lang="es"):
        """Open Google Translate"""
        try:
            translate_url = f"https://translate.google.com/?sl=auto&tl={target_lang}&text={urllib.parse.quote(text)}"
            self.open_google_chrome(translate_url)
            print(f"🌍 Google Translate: {text} -> {target_lang}")
            return True
        except Exception as e:
            print(f"Error opening Google Translate: {e}")
            return False
    
    def open_google_service(self, service):
        """Open various Google services"""
        services = {
            'gmail': 'https://mail.google.com',
            'drive': 'https://drive.google.com',
            'docs': 'https://docs.google.com',
            'sheets': 'https://sheets.google.com',
            'calendar': 'https://calendar.google.com',
            'maps': 'https://maps.google.com',
            'youtube': 'https://youtube.com',
            'meet': 'https://meet.google.com'
        }
        
        url = services.get(service.lower())
        if url:
            self.open_google_chrome(url)
            print(f"📧 Opened Google {service.title()}")
            return True
        else:
            print(f"Unknown Google service: {service}")
            return False
    
    def listen_for_voice_command(self):
        """Listen for voice command using Google Speech Recognition"""
        if not self.voice_recognition_enabled:
            print("Voice recognition not available")
            return None
        
        try:
            with self.microphone as source:
                print("🎤 Listening for voice command...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
            
            print("🧠 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"✓ Recognized: '{text}'")
            return text.lower()
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Google Speech Recognition error: {e}")
            return None
        except Exception as e:
            print(f"Error in voice recognition: {e}")
            return None
    
    def process_voice_command(self, command):
        """Process voice command for Google actions"""
        if not command:
            return False
        
        command = command.lower()
        
        # Google Search commands
        if "search" in command or "google" in command:
            # Extract search query
            if "search for" in command:
                query = command.split("search for", 1)[1].strip()
            elif "google" in command:
                query = command.split("google", 1)[1].strip()
            else:
                query = command
            
            if query:
                self.google_search(query)
                return True
        
        # Google Translate commands
        elif "translate" in command:
            # Extract text to translate
            parts = command.split("translate", 1)
            if len(parts) > 1:
                text = parts[1].strip()
                self.google_translate(text)
                return True
        
        # Open Google services
        elif "open" in command:
            for service in ['gmail', 'drive', 'docs', 'sheets', 'calendar', 'maps', 'youtube', 'meet']:
                if service in command:
                    self.open_google_service(service)
                    return True
        
        # Direct service names
        elif any(word in command for word in ['gmail', 'drive', 'docs', 'sheets', 'calendar', 'maps', 'youtube', 'meet']):
            for service in ['gmail', 'drive', 'docs', 'sheets', 'calendar', 'maps', 'youtube', 'meet']:
                if service in command:
                    self.open_google_service(service)
                    return True
        
        return False
    
    def detect_eye_gesture(self, current_time):
        """Detect eye-based gestures using blink patterns for volume control"""
        if not self.gesture_enabled or not self.volume_available:
            return None
        
        # Prevent rapid volume changes
        if current_time - self.last_volume_action_time < self.volume_action_cooldown:
            return None
        
        try:
            # Clean old blinks from buffer (outside time window)
            self.blink_pattern_buffer = [t for t in self.blink_pattern_buffer 
                                         if current_time - t < self.pattern_window]
            
            # Detect blink patterns
            num_blinks = len(self.blink_pattern_buffer)
            
            if num_blinks == 0:
                return None
            
            # Show current buffer status for debugging
            if num_blinks > 0:
                print(f"[VOLUME] Buffer: {num_blinks} blinks")
            
            # Check for triple blink first (highest priority)
            if num_blinks >= 3:
                # Triple blink = Volume DOWN
                # Check if first 3 blinks are close together
                time_span = self.blink_pattern_buffer[2] - self.blink_pattern_buffer[0]
                print(f"[VOLUME] ⭐ Triple blink! {num_blinks} blinks in {time_span:.2f}s")
                
                if time_span <= 2.0:  # All 3 blinks within 2 seconds (very lenient)
                    print(f"[VOLUME] ✅ VOLUME DOWN triggered!")
                    self.blink_pattern_buffer.clear()
                    self.last_volume_action_time = current_time
                    return "volume_down"
                else:
                    print(f"[VOLUME] ❌ Triple blink too slow ({time_span:.2f}s > 2.0s)")
            
            # Check for double blink (wait a short time to ensure it's not triple)
            elif num_blinks == 2:
                # Double blink = Volume UP
                time_span = self.blink_pattern_buffer[1] - self.blink_pattern_buffer[0]
                time_since_last = current_time - self.blink_pattern_buffer[-1]
                
                print(f"[VOLUME] ⭐ Double blink check: span={time_span:.2f}s, wait={time_since_last:.2f}s")
                
                # Quick double blink
                if time_span <= 1.5:  # Blinks within 1.5 seconds (more lenient)
                    # Shorter wait to make it more responsive
                    if time_since_last >= self.pattern_check_delay:  # Wait only 0.4s
                        print(f"[VOLUME] ✅ VOLUME UP triggered!")
                        self.blink_pattern_buffer.clear()
                        self.last_volume_action_time = current_time
                        return "volume_up"
                    else:
                        print(f"[VOLUME] ⏳ Waiting {self.pattern_check_delay - time_since_last:.1f}s more...")
                else:
                    print(f"[VOLUME] ❌ Double blink too slow ({time_span:.2f}s > 1.5s)")
            
            # Single blink - just show for debugging
            elif num_blinks == 1:
                time_since_last = current_time - self.blink_pattern_buffer[-1]
                if time_since_last < 1.0:
                    print(f"[VOLUME] Single blink detected, waiting for more...")
            
            return None
            
        except Exception as e:
            print(f"❌ Eye gesture detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_long_blink(self, current_time):
        """Detect long blink (hold) for mute/unmute"""
        if not self.volume_available:
            return None
            
        # Check if eyes have been closed for long time
        if self.eyes_closed_frames >= 15:  # About 0.5 seconds at 30fps
            if current_time - self.last_volume_action_time >= self.volume_action_cooldown:
                print(f"[VOLUME] Long blink detected: {self.eyes_closed_frames} frames")
                self.last_volume_action_time = current_time
                self.eyes_closed_frames = 0  # Reset to prevent repeat
                return "mute"
        return None
    
    def detect_one_eye_closed(self, num_eyes, current_time):
        """Detect SUSTAINED one eye closed (wink) for volume down - NOT a blink"""
        if not self.volume_available:
            return False
        
        # Track number of eyes detected
        self.one_eye_history.append(num_eyes)
        
        # Need enough history
        if len(self.one_eye_history) < self.one_eye_threshold:
            return False
        
        # Check cooldown
        if current_time - self.last_one_eye_volume_time < self.one_eye_volume_cooldown:
            return False
        
        # Get recent frames for analysis
        recent_frames = list(self.one_eye_history)
        
        # STRICT VALIDATION: Must be SUSTAINED single eye - NOT a blink!
        # During a blink, we might briefly see 1 eye during the transition
        # A true one-eye-closed (wink) should NEVER have 0 eyes
        
        # Check last 10 frames for pattern
        last_10 = recent_frames[-10:] if len(recent_frames) >= 10 else recent_frames
        
        # RULE 1: Must have NO zero-eye frames (distinguishes from blink)
        zero_eye_frames = sum(1 for count in last_10 if count == 0)
        if zero_eye_frames > 0:
            return False  # This is a blink, not one eye closed
        
        # RULE 2: Must have sustained consecutive 1-eye frames
        consecutive_one_eye = 0
        max_consecutive = 0
        for count in reversed(last_10):  # Check from most recent
            if count == 1:
                consecutive_one_eye += 1
                max_consecutive = max(max_consecutive, consecutive_one_eye)
            else:
                break  # Stop at first non-1 value
        
        # Require at least 8 consecutive frames with exactly 1 eye (about 0.27 seconds)
        # This ensures it's a deliberate wink, not a brief detection during blink
        sustained_wink = consecutive_one_eye >= 8
        
        # RULE 3: The previous state should have been 2 eyes (stable baseline)
        had_both_eyes_before = False
        if len(recent_frames) >= 12:
            # Check frames before the one-eye period
            frames_before = recent_frames[-12:-10]
            had_both_eyes_before = all(count == 2 for count in frames_before)
        
        if sustained_wink and had_both_eyes_before:
            print(f"[VOLUME] Sustained WINK detected: {consecutive_one_eye} consecutive frames with 1 eye")
            self.last_one_eye_volume_time = current_time
            # Clear history to prevent immediate re-trigger
            self.one_eye_history.clear()
            return True
        
        return False
    
    def change_volume(self, direction):
        """Change system volume"""
        if not self.volume_available:
            print("❌ Volume control not available - pycaw may not be installed")
            print("Install with: pip install pycaw comtypes")
            self.volume_change_feedback = "Volume Control Unavailable"
            self.volume_feedback_time = time.time()
            return
            
        try:
            if direction == "up":
                current_volume = self.volume.GetMasterVolumeLevel()
                new_volume = min(current_volume + 2.5, self.max_volume)
                self.volume.SetMasterVolumeLevel(new_volume, None)
                percentage = self.get_volume_percentage()
                print(f"\n🔊👁️👁️ VOLUME UP: {percentage:.0f}%")
                self.volume_change_feedback = f"Volume UP: {percentage:.0f}%"
                self.volume_feedback_time = time.time()
                
            elif direction == "down":
                current_volume = self.volume.GetMasterVolumeLevel()
                new_volume = max(current_volume - 2.5, self.min_volume)
                self.volume.SetMasterVolumeLevel(new_volume, None)
                percentage = self.get_volume_percentage()
                print(f"\n🔉👁️👁️👁️ VOLUME DOWN: {percentage:.0f}%")
                self.volume_change_feedback = f"Volume DOWN: {percentage:.0f}%"
                self.volume_feedback_time = time.time()
                
            elif direction == "mute":
                is_muted = self.volume.GetMute()
                self.volume.SetMute(not is_muted, None)
                status = 'UNMUTED' if is_muted else 'MUTED'
                print(f"\n🔇👁️━━ Volume {status}")
                self.volume_change_feedback = f"Volume {status}"
                self.volume_feedback_time = time.time()
                
        except Exception as e:
            print(f"❌ Volume control error: {e}")
            print("Make sure pycaw and comtypes are installed")
            self.volume_available = False
    
    def get_volume_percentage(self):
        """Get current volume as percentage"""
        try:
            current_volume = self.volume.GetMasterVolumeLevel()
            volume_range = self.max_volume - self.min_volume
            percentage = ((current_volume - self.min_volume) / volume_range) * 100
            return percentage
        except:
            return 0
    
    def start_calibration(self):
        """Start enhanced interactive calibration process"""
        self.calibration_mode = True
        self.calibration_complete = False
        self.current_calibration_point = 0
        self.calibration_samples = []
        self.current_point_samples = []
        self.calibration_start_time = None
        self.calibration_gaze_data = []
        # Reset calibration data
        self.calibration_data = {
            'min_x': float('inf'),
            'max_x': float('-inf'),
            'min_y': float('inf'),
            'max_y': float('-inf')
        }
        self.frames_processed = 0
        self.smooth_buffer.clear()
        self.gaze_buffer.clear()
        print("\n" + "="*60)
        print("🎯 ENHANCED 9-POINT CALIBRATION STARTED")
        print("="*60)
        print("Look at each GREEN circle and hold your gaze steady.")
        print("The circle will turn BLUE while collecting data.")
        print("Follow all 9 calibration points for maximum accuracy.")
        print("Cover corners, edges, and center of your screen.")
        print("Machine learning model will be trained automatically.")
        print("="*60 + "\n")
    
    def finish_calibration(self):
        """Complete calibration with ML-based model training and validation"""
        if len(self.calibration_gaze_data) >= 9:
            # Build calibration ranges from collected data
            all_x = [g[0] for g in self.calibration_gaze_data]
            all_y = [g[1] for g in self.calibration_gaze_data]
            
            self.calibration_data['min_x'] = min(all_x)
            self.calibration_data['max_x'] = max(all_x)
            self.calibration_data['min_y'] = min(all_y)
            self.calibration_data['max_y'] = max(all_y)
            
            # Train polynomial regression model for gaze mapping
            # This creates a mapping from eye position to screen coordinates
            self.train_gaze_mapping_model()
            
            self.frames_processed = 100
            self.calibration_complete = True
            
            # Calculate calibration quality metrics
            x_range = self.calibration_data['max_x'] - self.calibration_data['min_x']
            y_range = self.calibration_data['max_y'] - self.calibration_data['min_y']
            
            # Calculate variance (consistency)
            x_variance = np.var(all_x)
            y_variance = np.var(all_y)
            total_variance = x_variance + y_variance
            
            # Calculate coverage (how much screen area is covered)
            coverage_score = min(100, (x_range / 200) * 50 + (y_range / 200) * 50)
            
            # Calculate consistency score (lower variance is better)
            consistency_score = max(0, 100 - (total_variance / 100))
            
            # Combined accuracy score
            self.calibration_accuracy_score = int((coverage_score * 0.6 + consistency_score * 0.4))
            
            # Validate calibration quality
            if self.calibration_accuracy_score >= 85:
                quality = "EXCELLENT ✓"
            elif self.calibration_accuracy_score >= 70:
                quality = "GOOD ✓"
            elif self.calibration_accuracy_score >= 55:
                quality = "FAIR"
            else:
                quality = "POOR - Recalibrate"
            
            print("\n" + "="*60)
            print("✓ CALIBRATION COMPLETE - MODEL TRAINED!")
            print("="*60)
            print(f"📊 9-Point Calibration Score: {self.calibration_accuracy_score}% ({quality})")
            print(f"📏 X Range: {self.calibration_data['min_x']:.1f} - {self.calibration_data['max_x']:.1f}")
            print(f"📏 Y Range: {self.calibration_data['min_y']:.1f} - {self.calibration_data['max_y']:.1f}")
            print(f"📐 Coverage: {coverage_score:.1f}%")
            print(f"🎯 Consistency: {consistency_score:.1f}%")
            print(f"🧠 Gaze Mapping Model: Polynomial Regression (Trained)")
            print("\n👁️  Eye tracking is now active - move your eyes!")
            print("\n�️  Eye Gesture Controls (Blink Patterns):")
            print("  👁️  Single Blink     = Click")
            print("  👁️👁️  Double Blink    = Volume UP")
            print("  👁️👁️👁️  Triple Blink   = Volume DOWN")
            print("  👁️━━  Long Blink (hold) = Mute/Unmute")
            print("  👁️  One Eye Closed  = Volume DOWN (hold one eye shut)")
            print("\n⌨️  Keyboard Controls:")
            print("  'R' = Recalibrate")
            print("  'F' = Toggle focus lock")
            print("  'P' = Toggle precision mode")
            print("  'Q' = Quit")
            print("="*60 + "\n")
            return True
        else:
            print("❌ Calibration failed - not enough data points")
            return False
    
    def train_gaze_mapping_model(self):
        """Train polynomial regression model for accurate gaze-to-screen mapping"""
        try:
            # Prepare training data: eye positions -> screen positions
            if len(self.calibration_samples) < 9:
                return
            
            # Extract features (eye gaze positions) and targets (screen positions)
            eye_positions_x = []
            eye_positions_y = []
            screen_positions_x = []
            screen_positions_y = []
            
            for sample in self.calibration_samples:
                if 'eye_pos' in sample and 'screen_pos' in sample:
                    eye_positions_x.append(sample['eye_pos'][0])
                    eye_positions_y.append(sample['eye_pos'][1])
                    screen_positions_x.append(sample['screen_pos'][0])
                    screen_positions_y.append(sample['screen_pos'][1])
            
            if len(eye_positions_x) < 9:
                return
            
            # Calculate polynomial coefficients for X and Y mapping
            # Use numpy polyfit for 2nd degree polynomial
            self.model_x_coeffs = np.polyfit(eye_positions_x, screen_positions_x, 2)
            self.model_y_coeffs = np.polyfit(eye_positions_y, screen_positions_y, 2)
            
            # Calculate R-squared score for model quality
            predicted_x = np.polyval(self.model_x_coeffs, eye_positions_x)
            predicted_y = np.polyval(self.model_y_coeffs, eye_positions_y)
            
            # R-squared calculation
            ss_res_x = np.sum((np.array(screen_positions_x) - predicted_x) ** 2)
            ss_tot_x = np.sum((np.array(screen_positions_x) - np.mean(screen_positions_x)) ** 2)
            r2_x = 1 - (ss_res_x / ss_tot_x) if ss_tot_x != 0 else 0
            
            ss_res_y = np.sum((np.array(screen_positions_y) - predicted_y) ** 2)
            ss_tot_y = np.sum((np.array(screen_positions_y) - np.mean(screen_positions_y)) ** 2)
            r2_y = 1 - (ss_res_y / ss_tot_y) if ss_tot_y != 0 else 0
            
            self.model_accuracy = (r2_x + r2_y) / 2
            
            print(f"🧠 Model Training Complete:")
            print(f"   X-axis R² score: {r2_x:.3f}")
            print(f"   Y-axis R² score: {r2_y:.3f}")
            print(f"   Overall accuracy: {self.model_accuracy*100:.1f}%")
            
        except Exception as e:
            print(f"Model training error: {e}")
            self.model_x_coeffs = None
            self.model_y_coeffs = None
            self.model_accuracy = 0
        
    def detect_pupil(self, eye_img):
        """Enhanced pupil detection with better accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        gray = cv2.GaussianBlur(gray, (7, 7), 2)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Multiple threshold approaches for robustness
        _, threshold1 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        threshold2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 3)
        
        # Combine thresholds
        threshold = cv2.bitwise_and(threshold1, threshold2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50 and area < eye_img.shape[0] * eye_img.shape[1] * 0.6:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:
                            valid_contours.append(contour)
            
            if valid_contours:
                # Get largest valid contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Use enclosing circle for better center estimation
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                
                if radius > 5 and radius < min(eye_img.shape[0], eye_img.shape[1]) / 2:
                    return (int(cx), int(cy), radius)
        
        return None
    
    def process_calibration_frame(self, frame):
        """Process frame during calibration with visual feedback"""
        frame_h, frame_w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw calibration UI
        cv2.putText(frame, "CALIBRATION MODE", (frame_w//2 - 180, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"Point {self.current_calibration_point + 1} of {len(self.calibration_points_targets)}",
                   (frame_w//2 - 120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Get current calibration target
        if self.current_calibration_point < len(self.calibration_points_targets):
            target = self.calibration_points_targets[self.current_calibration_point]
            target_x = int(target[0] * frame_w)
            target_y = int(target[1] * frame_h)
            
            # Draw calibration target
            is_collecting = len(self.calibration_samples) > 0
            color = (255, 0, 0) if is_collecting else (0, 255, 0)
            
            # Animated pulsing circle
            pulse = int(5 * abs(np.sin(time.time() * 3)))
            cv2.circle(frame, (target_x, target_y), 40 + pulse, color, 3)
            cv2.circle(frame, (target_x, target_y), 20, color, -1)
            cv2.circle(frame, (target_x, target_y), 8, (255, 255, 255), -1)
            
            # Draw crosshair
            cv2.line(frame, (target_x - 50, target_y), (target_x + 50, target_y), color, 2)
            cv2.line(frame, (target_x, target_y - 50), (target_x, target_y + 50), color, 2)
            
            # Progress bar
            if is_collecting:
                progress = len(self.calibration_samples) / self.samples_per_point
                bar_width = 300
                bar_height = 30
                bar_x = frame_w//2 - bar_width//2
                bar_y = frame_h - 80
                
                # Background
                cv2.rectangle(frame, (bar_x - 2, bar_y - 2), 
                            (bar_x + bar_width + 2, bar_y + bar_height + 2), (255, 255, 255), 2)
                # Progress
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                # Text
                cv2.putText(frame, "Collecting data - Keep looking at the target!", 
                           (bar_x - 50, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Show instructions
                cv2.putText(frame, "Look at the GREEN target", 
                           (frame_w//2 - 180, frame_h - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Process eye tracking
            processed = self.preprocess_for_detection(gray)
            faces = self.face_cascade.detectMultiScale(processed, 1.2, 4, minSize=(100, 100))
            
            eyes_detected = False
            avg_gaze_x = 0
            avg_gaze_y = 0
            
            for (x, y, w, h) in faces:
                eye_region_y = y + int(h * 0.2)
                eye_region_h = int(h * 0.4)
                
                roi_gray = gray[eye_region_y:eye_region_y+eye_region_h, x:x+w]
                roi_color = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w]
                
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(30, 30))
                
                eyes_data = []
                for (ex, ey, ew, eh) in eyes:
                    # Draw eye rectangle
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                    pupil = self.detect_pupil(eye_img)
                    
                    if pupil:
                        pupil_x = ex + pupil[0]
                        pupil_y = ey + pupil[1]
                        cv2.circle(roi_color, (pupil_x, pupil_y), 3, (0, 0, 255), -1)
                        
                        abs_x = x + ex + pupil[0]
                        abs_y = y + ey + pupil[1]
                        eyes_data.append((abs_x, abs_y))
                    else:
                        # Use eye center as fallback
                        eye_center_x = ex + ew // 2
                        eye_center_y = ey + eh // 2
                        cv2.circle(roi_color, (eye_center_x, eye_center_y), 5, (255, 0, 255), -1)
                        
                        abs_x = x + eye_center_x
                        abs_y = y + eye_center_y
                        eyes_data.append((abs_x, abs_y))
                
                if len(eyes_data) >= 1:
                    avg_gaze_x = sum([e[0] for e in eyes_data]) / len(eyes_data)
                    avg_gaze_y = sum([e[1] for e in eyes_data]) / len(eyes_data)
                    eyes_detected = True
            
            # Calibration logic
            if eyes_detected:
                current_time = time.time()
                
                if self.calibration_start_time is None:
                    self.calibration_start_time = current_time
                
                # Get current target screen position
                target = self.calibration_points_targets[self.current_calibration_point]
                screen_x = int(target[0] * self.screen_w)
                screen_y = int(target[1] * self.screen_h)
                
                # Wait for stabilization, then collect
                if current_time - self.calibration_start_time >= self.calibration_delay:
                    # Store both eye position and corresponding screen position
                    sample = {
                        'eye_pos': (avg_gaze_x, avg_gaze_y),
                        'screen_pos': (screen_x, screen_y)
                    }
                    self.current_point_samples.append(sample)
                    self.calibration_samples.append(sample)  # Also add to master list
                    
                    # Move to next point when enough samples collected
                    if len(self.current_point_samples) >= self.samples_per_point:
                        # Average eye positions for this point
                        final_eye_x = sum([s['eye_pos'][0] for s in self.current_point_samples]) / len(self.current_point_samples)
                        final_eye_y = sum([s['eye_pos'][1] for s in self.current_point_samples]) / len(self.current_point_samples)
                        
                        self.calibration_gaze_data.append((final_eye_x, final_eye_y))
                        print(f"✓ Calibration point {self.current_calibration_point + 1} recorded: Eye({final_eye_x:.1f}, {final_eye_y:.1f}) -> Screen({screen_x}, {screen_y})")
                        
                        # Next point
                        self.current_calibration_point += 1
                        self.current_point_samples = []  # Reset for next point
                        self.calibration_start_time = None
                        
                        # Check if done
                        if self.current_calibration_point >= len(self.calibration_points_targets):
                            self.calibration_mode = False
                            self.finish_calibration()
                else:
                    # Countdown
                    remaining = self.calibration_delay - (current_time - self.calibration_start_time)
                    cv2.putText(frame, f"Hold steady... {remaining:.1f}s", 
                               (frame_w//2 - 150, frame_h - 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                # Reset timer if eyes not detected
                self.calibration_start_time = None
                cv2.putText(frame, "Eyes not detected - Face the camera", 
                           (frame_w//2 - 250, frame_h - 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def smooth_position(self, x, y):
        """Advanced smoothing with weighted averaging and exponential smoothing"""
        self.smooth_buffer.append((x, y))
        if len(self.smooth_buffer) == 0:
            return x, y
        
        # Weighted averaging with Gaussian-like weights for smoother tracking
        weights = np.exp(np.linspace(-0.5, 1, len(self.smooth_buffer)))
        weights = weights / weights.sum()
        
        avg_x = sum([p[0] * w for p, w in zip(self.smooth_buffer, weights)])
        avg_y = sum([p[1] * w for p, w in zip(self.smooth_buffer, weights)])
        
        # Apply exponential smoothing for additional stability
        if self.exponential_smoothing and self.last_cursor_pos:
            avg_x = self.last_cursor_pos[0] * (1 - self.alpha) + avg_x * self.alpha
            avg_y = self.last_cursor_pos[1] * (1 - self.alpha) + avg_y * self.alpha
        
        # Adaptive dead zone based on movement speed
        current_dead_zone = self.dead_zone
        if self.adaptive_dead_zone and self.last_cursor_pos:
            dx = abs(avg_x - self.last_cursor_pos[0])
            dy = abs(avg_y - self.last_cursor_pos[1])
            movement = np.sqrt(dx*dx + dy*dy)
            
            # Larger dead zone for small movements (reduces jitter), smaller for intentional movements
            if movement < 8:
                current_dead_zone = self.dead_zone * 2.0  # Much more stable for small jitter
            elif movement > 100:
                current_dead_zone = self.dead_zone * 0.4
            
            # If movement is within dead zone, keep previous position
            if dx < current_dead_zone and dy < current_dead_zone:
                return self.last_cursor_pos[0], self.last_cursor_pos[1]
        
        # Fine tune mode - reduce movement for precision
        if self.fine_tune_mode:
            if self.last_cursor_pos:
                diff_x = avg_x - self.last_cursor_pos[0]
                diff_y = avg_y - self.last_cursor_pos[1]
                avg_x = self.last_cursor_pos[0] + diff_x * 0.2
                avg_y = self.last_cursor_pos[1] + diff_y * 0.2
        
        smooth_x, smooth_y = int(avg_x), int(avg_y)
        self.last_cursor_pos = (smooth_x, smooth_y)
        
        return smooth_x, smooth_y
    
    def detect_blink(self, num_eyes):
        """Fast and accurate blink detection with multiple patterns"""
        self.eyes_detected_history.append(num_eyes)
        
        current_time = time.time()
        
        # Track eyes closed/open frames for long blink detection
        if num_eyes == 0:
            self.eyes_closed_frames += 1
            self.eyes_open_frames = 0
        else:
            self.eyes_open_frames += 1
            if self.eyes_open_frames >= 2:
                self.eyes_closed_frames = 0
        
        # Need minimal history for fast response
        if len(self.eyes_detected_history) < 5:
            return False
        
        # Check cooldown
        if current_time - self.last_blink_time < self.blink_cooldown:
            return False
        
        # Convert to list for easier analysis
        history = list(self.eyes_detected_history)
        
        # Multiple detection patterns - OPTIMIZED FOR SPEED AND ACCURACY
        blink_detected = False
        
        # Pattern 1: ULTRA-FAST BLINK (1 frame closure) - Most Responsive
        if len(history) >= 5:
            # Eyes present -> single frame closure -> eyes back immediately
            had_eyes = history[-5] >= 2
            single_closure = history[-4] == 0 or history[-3] == 0
            eyes_back = history[-2] >= 1 and history[-1] >= 1
            
            if had_eyes and single_closure and eyes_back:
                blink_detected = True
                self.blink_confidence = 0.80
                print("[BLINK] Pattern 1: Ultra-fast blink (1 frame)")
        
        # Pattern 2: QUICK BLINK (2 frame closure) - Very Common
        if not blink_detected and len(history) >= 6:
            # Eyes present -> 2 frame closure -> eyes back
            had_eyes = history[-6] >= 2 or history[-5] >= 2
            double_closure = history[-4] == 0 and history[-3] == 0
            eyes_back = history[-2] >= 1 and history[-1] >= 1
            
            if had_eyes and double_closure and eyes_back:
                blink_detected = True
                self.blink_confidence = 0.90
                print("[BLINK] Pattern 2: Quick blink (2 frames)")
        
        # Pattern 3: STANDARD BLINK (2-3 frame closure)
        if not blink_detected and len(history) >= 6:
            # Any eyes -> sustained closure -> any eyes back
            had_eyes = history[-6] >= 1 or history[-5] >= 1
            closure = (history[-4] == 0 and history[-3] == 0) or (history[-3] == 0 and history[-2] == 0)
            eyes_back = history[-1] >= 1
            
            if had_eyes and closure and eyes_back:
                blink_detected = True
                self.blink_confidence = 0.85
                print("[BLINK] Pattern 3: Standard blink (2-3 frames)")
        
        # Pattern 4: MEDIUM BLINK (3-4 frames) with validation
        if not blink_detected and len(history) >= 7:
            # Eyes present -> longer closure -> clear recovery
            had_eyes = history[-7] >= 1 and history[-6] >= 1
            longer_closure = history[-5] == 0 and history[-4] == 0 and (history[-3] == 0 or history[-2] == 0)
            clear_recovery = history[-1] >= 1
            
            if had_eyes and longer_closure and clear_recovery:
                blink_detected = True
                self.blink_confidence = 0.92
                print("[BLINK] Pattern 4: Medium blink (3-4 frames)")
        
        # Pattern 5: LENIENT - catch any closure pattern
        if not blink_detected and len(history) >= 5:
            # Simple pattern: had eyes -> lost eyes -> got eyes back
            transitions = []
            for i in range(len(history) - 1):
                if history[i] >= 1 and history[i+1] == 0:
                    transitions.append('close')
                elif history[i] == 0 and history[i+1] >= 1:
                    transitions.append('open')
            
            # If we see a close followed by open in recent history = blink
            if 'close' in transitions[-4:] and 'open' in transitions[-3:]:
                # Validate: current state should be eyes open
                if history[-1] >= 1:
                    blink_detected = True
                    self.blink_confidence = 0.75
                    print("[BLINK] Pattern 5: General blink pattern detected")
        
        # SMART FILTERING: Only reject obvious false positives
        if blink_detected:
            # Only reject if it looks like sustained single-eye (wink)
            recent = history[-6:] if len(history) >= 6 else history
            if recent.count(1) >= 5:  # Almost all frames showing 1 eye = sustained wink
                print("[BLINK] Rejected: Sustained single-eye (wink)")
                return False
            
            # Accept the blink - FAST RESPONSE
            self.last_blink_time = current_time
            self.blink_detected_frame = 5
            print(f"[BLINK] ✓ DETECTED! (confidence: {self.blink_confidence:.0%})")
            return True
        
        return False
    
    def update_calibration(self, x, y):
        """Auto-calibrate by tracking min/max positions"""
        self.calibration_data['min_x'] = min(self.calibration_data['min_x'], x)
        self.calibration_data['max_x'] = max(self.calibration_data['max_x'], x)
        self.calibration_data['min_y'] = min(self.calibration_data['min_y'], y)
        self.calibration_data['max_y'] = max(self.calibration_data['max_y'], y)
    
    def map_to_screen(self, gaze_x, gaze_y):
        """Enhanced mapping with ML model and sensitivity control"""
        if self.frames_processed < 100:
            # Still calibrating
            return None, None
        
        # Use trained polynomial model if available
        if self.model_x_coeffs is not None and self.model_y_coeffs is not None:
            # Apply polynomial mapping
            screen_x = np.polyval(self.model_x_coeffs, gaze_x)
            screen_y = np.polyval(self.model_y_coeffs, gaze_y)
            
            # Apply sensitivity adjustment
            center_x, center_y = self.screen_w / 2, self.screen_h / 2
            screen_x = center_x + (screen_x - center_x) * self.sensitivity
            screen_y = center_y + (screen_y - center_y) * self.sensitivity
            
            # Constrain to screen bounds
            screen_x = max(0, min(int(screen_x), self.screen_w - 1))
            screen_y = max(0, min(int(screen_y), self.screen_h - 1))
            
            return screen_x, screen_y
        
        # Fallback to linear mapping if model not trained
        # Normalize based on calibration
        x_range = self.calibration_data['max_x'] - self.calibration_data['min_x']
        y_range = self.calibration_data['max_y'] - self.calibration_data['min_y']
        
        if x_range > 0:
            norm_x = (gaze_x - self.calibration_data['min_x']) / x_range
        else:
            norm_x = 0.5
        
        if y_range > 0:
            norm_y = (gaze_y - self.calibration_data['min_y']) / y_range
        else:
            norm_y = 0.5
        
        # Apply sensitivity and center around 0.5
        norm_x = 0.5 + (norm_x - 0.5) * self.sensitivity
        norm_y = 0.5 + (norm_y - 0.5) * self.sensitivity
        
        # Map to screen with enhanced padding for edge access
        padding_factor = 1.3
        screen_x = int(norm_x * self.screen_w * padding_factor - self.screen_w * (padding_factor - 1) / 2)
        screen_y = int(norm_y * self.screen_h * padding_factor - self.screen_h * (padding_factor - 1) / 2)
        
        # Constrain to screen bounds
        screen_x = max(0, min(screen_x, self.screen_w - 1))
        screen_y = max(0, min(screen_y, self.screen_h - 1))
        
        return screen_x, screen_y
    
    def process_frame(self, frame):
        """Process frame with eye tracking and eye-based gesture detection"""
        frame_h, frame_w, _ = frame.shape
        
        # Handle calibration mode
        if self.calibration_mode:
            return self.process_calibration_frame(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple preprocessing
        processed = self.preprocess_for_detection(gray)
        
        # Single face detection pass with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            processed, 
            scaleFactor=1.2, 
            minNeighbors=4,
            minSize=(100, 100)
        )
        
        # Use last known face position if no face detected
        if len(faces) == 0 and self.last_face_rect is not None:
            self.face_miss_counter += 1
            if self.face_miss_counter < self.max_face_misses:
                faces = [self.last_face_rect]
        else:
            self.face_miss_counter = 0
            if len(faces) > 0:
                self.last_face_rect = faces[0]  # Store first face
        
        eyes_data = []
        num_eyes_detected = 0
        
        # Store for external access (web interface)
        self.num_eyes_detected = 0
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Eye region (upper 50% of face)
            eye_region_y = y + int(h * 0.2)
            eye_region_h = int(h * 0.4)
            
            roi_gray = gray[eye_region_y:eye_region_y+eye_region_h, x:x+w]
            roi_color = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w]
            
            # BALANCED eye detection - strict enough to reject closed eyes, lenient enough for open eyes
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.12,  # Slightly reduced for better detection
                minNeighbors=7,  # Balanced - not too strict, not too lenient
                minSize=(30, 30),  # Allow smaller eyes
                maxSize=(80, 80)  # Reasonable max size
            )
            
            # SMART FILTERING: validate eye detections with balanced criteria
            filtered_eyes = []
            for (ex, ey, ew, eh) in eyes:
                eye_patch = roi_gray[ey:ey+eh, ex:ex+ew]
                if eye_patch.size == 0:
                    continue
                
                # TEST 1: Aspect ratio (eyes should be wider than tall)
                aspect_ratio = ew / eh if eh > 0 else 0
                if not (0.9 < aspect_ratio < 3.0):  # Reasonable range
                    continue
                
                # TEST 2: Contrast check (open eyes have contrast due to pupil)
                contrast = np.std(eye_patch)
                if contrast < 15:  # Open eyes have pupil contrast
                    continue
                
                # TEST 3: Intensity range (pupil vs sclera difference)
                min_val = np.min(eye_patch)
                max_val = np.max(eye_patch)
                intensity_range = max_val - min_val
                if intensity_range < 25:  # Must have some range
                    continue
                
                # Passed basic tests - likely an eye region
                filtered_eyes.append((ex, ey, ew, eh))
            
            # IMPROVED: Remove duplicate detections of the same eye
            # Group eyes that are too close together (likely duplicates)
            unique_eyes = []
            for eye in sorted(filtered_eyes, key=lambda e: e[0]):
                ex, ey, ew, eh = eye
                is_duplicate = False
                for unique_eye in unique_eyes:
                    uex, uey, uew, ueh = unique_eye
                    # Calculate center distance
                    center_dist = abs((ex + ew/2) - (uex + uew/2))
                    # If centers are within 50% of eye width, consider duplicate
                    if center_dist < ew * 0.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_eyes.append(eye)
            
            # Take first 2 unique eyes only, sorted left to right
            valid_eyes = unique_eyes[:2]
            
            for (ex, ey, ew, eh) in valid_eyes:
                # SMART VALIDATION: Confirm this is an open eye (not closed)
                eye_patch = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Check for pupil-like dark region
                is_open_eye = False
                if eye_patch.size > 0:
                    mean_intensity = np.mean(eye_patch)
                    std_intensity = np.std(eye_patch)
                    
                    # Look for dark pixels (pupil area)
                    dark_threshold = mean_intensity - std_intensity * 0.7  # More lenient threshold
                    dark_pixels = np.sum(eye_patch < dark_threshold)
                    total_pixels = eye_patch.size
                    dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # Open eye criteria (more relaxed for single eye detection):
                    # - Has dark region (pupil) OR good contrast OR reasonable intensity
                    has_dark_region = dark_ratio > 0.05  # Reduced from 0.08 to 0.05
                    has_contrast = std_intensity > 10  # Reduced from 12 to 10
                    has_reasonable_intensity = mean_intensity < 200  # Not too bright (closed eyelid is often bright)
                    
                    # Accept if any two criteria met (more lenient)
                    criteria_met = sum([has_dark_region, has_contrast, has_reasonable_intensity])
                    is_open_eye = criteria_met >= 2
                
                # Count if it looks like an open eye
                if is_open_eye:
                    num_eyes_detected += 1
                    self.num_eyes_detected = num_eyes_detected  # Update instance variable
                    
                    # Draw eye rectangle
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Extract eye region
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                    
                    # Detect pupil
                    pupil = self.detect_pupil(eye_img)
                    
                    if pupil:
                        # Draw pupil with radius
                        pupil_x = ex + pupil[0]
                        pupil_y = ey + pupil[1]
                        pupil_radius = int(pupil[2]) if len(pupil) > 2 else 5
                        
                        # Draw pupil center and circle
                        cv2.circle(roi_color, (pupil_x, pupil_y), pupil_radius, (0, 255, 255), 2)
                        cv2.circle(roi_color, (pupil_x, pupil_y), 3, (0, 0, 255), -1)
                        
                        # Calculate relative position in eye
                        rel_x = pupil[0] / ew
                        rel_y = pupil[1] / eh
                        
                        # Store eye data (absolute position on frame)
                        abs_x = x + ex + pupil[0]
                        abs_y = y + ey + pupil[1]
                        eyes_data.append((abs_x, abs_y, rel_x, rel_y))
                    else:
                        # Fallback: use eye center if pupil detection fails
                        eye_center_x = ex + ew // 2
                        eye_center_y = ey + eh // 2
                        
                        # Draw eye center marker
                        cv2.circle(roi_color, (eye_center_x, eye_center_y), 5, (255, 0, 255), -1)
                        
                        # Use eye center position
                        rel_x = 0.5
                        rel_y = 0.5
                        
                        abs_x = x + eye_center_x
                        abs_y = y + eye_center_y
                        eyes_data.append((abs_x, abs_y, rel_x, rel_y))
        
        # Calculate average gaze position from both eyes
        if len(eyes_data) >= 1:
            avg_x = sum([e[0] for e in eyes_data]) / len(eyes_data)
            avg_y = sum([e[1] for e in eyes_data]) / len(eyes_data)
            
            # Store in gaze buffer for stability check
            self.gaze_buffer.append((avg_x, avg_y))
            
            # Calculate tracking confidence based on consistency
            if len(self.gaze_buffer) >= 3:
                positions = list(self.gaze_buffer)
                variance = np.var([p[0] for p in positions]) + np.var([p[1] for p in positions])
                self.tracking_confidence = min(100, max(0, 100 - variance / 5))
            else:
                self.tracking_confidence = 50
            
            # Update calibration only if interactive calibration not complete
            if not self.calibration_complete:
                self.update_calibration(avg_x, avg_y)
                self.frames_processed += 1
            
            # Map to screen coordinates
            screen_x, screen_y = self.map_to_screen(avg_x, avg_y)
            
            if screen_x is not None:
                # Apply smoothing
                screen_x, screen_y = self.smooth_position(screen_x, screen_y)
                
                # Focus locking logic
                if self.focus_lock_enabled:
                    if self.last_cursor_pos:
                        dx = abs(screen_x - self.last_cursor_pos[0])
                        dy = abs(screen_y - self.last_cursor_pos[1])
                        movement = np.sqrt(dx*dx + dy*dy)
                        
                        if movement < 20:  # Gaze is steady
                            if self.focus_lock_start_time is None:
                                self.focus_lock_start_time = time.time()
                            elif time.time() - self.focus_lock_start_time > self.focus_lock_threshold:
                                if not self.focus_locked:
                                    self.focus_locked = True
                                    self.locked_position = (screen_x, screen_y)
                                    print("Focus LOCKED at", self.locked_position)
                        else:
                            # Movement detected, unlock
                            if self.focus_locked:
                                print("Focus UNLOCKED")
                            self.focus_locked = False
                            self.focus_lock_start_time = None
                            self.locked_position = None
                    
                    # Use locked position if locked
                    if self.focus_locked and self.locked_position:
                        screen_x, screen_y = self.locked_position
                
                # Check for button hover
                current_button = self.check_button_hover(screen_x, screen_y)
                current_time = time.time()
                
                if current_button:
                    if self.hovered_button != current_button:
                        # New button being hovered
                        self.hovered_button = current_button
                        self.button_hover_time = current_time
                        self.frozen_cursor_pos = (screen_x, screen_y)
                    else:
                        # Same button - freeze cursor at initial hover position
                        if self.cursor_freeze_on_hover and self.frozen_cursor_pos:
                            screen_x, screen_y = self.frozen_cursor_pos
                        
                        # Check if hover time exceeded
                        hover_duration = current_time - self.button_hover_time
                        if hover_duration >= self.button_click_threshold:
                            # Auto-click button after hovering
                            self.execute_button_action(current_button['action'])
                            self.hovered_button = None
                            self.frozen_cursor_pos = None
                else:
                    # Not hovering over any button
                    self.hovered_button = None
                    self.frozen_cursor_pos = None
                
                # Control cursor with speed multiplier
                if self.control_enabled:
                    try:
                        # Slower, smoother movement
                        duration = 0.05 * (1 / self.speed_multiplier)
                        if self.fine_tune_mode:
                            duration = 0.1  # Even slower for precision
                        if self.hovered_button and self.cursor_freeze_on_hover:
                            duration = 0.02  # Quick snap when hovering on button
                        pyautogui.moveTo(screen_x, screen_y, duration=duration)
                    except:
                        pass
                
                # Store as last valid gaze
                self.last_valid_gaze = (screen_x, screen_y)
                
                # Enhanced visual feedback
                cv2.putText(frame, f"Cursor: ({screen_x}, {screen_y})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {int(self.tracking_confidence)}%", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Focus lock indicator
                if self.focus_lock_enabled:
                    if self.focus_locked:
                        cv2.putText(frame, "FOCUS LOCKED", (10, 180),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    elif self.focus_lock_start_time:
                        progress = (time.time() - self.focus_lock_start_time) / self.focus_lock_threshold
                        cv2.putText(frame, f"Locking... {int(progress*100)}%", (10, 180),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Fine tune mode indicator
                if self.fine_tune_mode:
                    cv2.putText(frame, "PRECISION MODE", (frame.shape[1] - 250, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Draw tracking indicator
                indicator_color = (0, 255, 0) if self.tracking_confidence > 70 else (0, 255, 255) if self.tracking_confidence > 40 else (0, 0, 255)
                cv2.circle(frame, (frame.shape[1] - 30, 30), 15, indicator_color, -1)
            else:
                progress = int((self.frames_processed / 100) * 100)
                cv2.putText(frame, f"CALIBRATING {progress}% - Look around!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display gaze position
            cal_status = "CALIBRATED" if self.calibration_complete else "NOT CALIBRATED"
            cal_color = (0, 255, 0) if self.calibration_complete else (0, 0, 255)
            
            # Add detection quality indicator
            detection_quality = "EXCELLENT" if len(eyes_data) >= 2 else "GOOD" if len(eyes_data) == 1 else "POOR"
            quality_color = (0, 255, 0) if len(eyes_data) >= 2 else (0, 255, 255) if len(eyes_data) == 1 else (0, 0, 255)
            
            blink_status = " | BLINK:ON" if self.blink_enabled else ""
            accuracy_text = f" | Accuracy:{self.calibration_accuracy_score}%" if self.calibration_complete else ""
            cv2.putText(frame, f"{cal_status} | Eyes: {num_eyes_detected} ({detection_quality}){blink_status}{accuracy_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_color, 2)
            
            # Show eye count history for debugging blink/wink detection
            if len(self.eyes_detected_history) > 0:
                recent_counts = list(self.eyes_detected_history)[-10:]
                history_text = "Eye History: " + " ".join(str(c) for c in recent_counts)
                cv2.putText(frame, history_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if not self.calibration_complete:
                cv2.putText(frame, ">>> Press 'R' to calibrate! <<<", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No eyes detected
            cv2.putText(frame, "NO EYES DETECTED - Adjust lighting/position", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Tips: Face the camera, ensure good lighting", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Detect blinks for clicking and volume control
        blink_happened = self.detect_blink(num_eyes_detected)
        
        if blink_happened:
            current_time = time.time()
            if self.control_enabled and self.blink_enabled:
                # Add to blink pattern buffer for gesture detection
                self.blink_pattern_buffer.append(current_time)
                print(f"[DEBUG] Blink detected! Buffer now has {len(self.blink_pattern_buffer)} blinks")
        
        # Eye gesture detection for volume control
        current_time = time.time()
        
        # Check for one eye closed (volume down)
        one_eye_closed = self.detect_one_eye_closed(num_eyes_detected, current_time)
        if one_eye_closed:
            self.change_volume("down")
            print(f"[DEBUG] One eye closed - Volume DOWN triggered")
            # Clear blink pattern buffer to avoid interference
            self.blink_pattern_buffer.clear()
        
        # Check for long blink (mute)
        long_blink_gesture = self.detect_long_blink(current_time)
        if long_blink_gesture == "mute":
            self.change_volume("mute")
            self.blink_pattern_buffer.clear()
        
        # Check for blink pattern gestures (double/triple blink)
        eye_gesture = self.detect_eye_gesture(current_time)
        if eye_gesture:
            if eye_gesture == "volume_up":
                self.change_volume("up")
                print(f"[DEBUG] Volume UP triggered")
            elif eye_gesture == "volume_down":
                self.change_volume("down")
                print(f"[DEBUG] Volume DOWN triggered")
        
        # Single blink = Click (only if pattern detection didn't consume it)
        # Check after a delay to allow pattern detection
        if blink_happened and len(self.blink_pattern_buffer) > 0:
            import threading
            def delayed_click():
                time.sleep(1.0)  # Wait for pattern to complete (increased)
                if len(self.blink_pattern_buffer) == 1:  # Still only one blink
                    try:
                        pyautogui.click()
                        print(f"👁️ Single blink - CLICK at {time.strftime('%H:%M:%S')}")
                    except:
                        pass
                    # Clear buffer after clicking
                    self.blink_pattern_buffer.clear()
            threading.Thread(target=delayed_click, daemon=True).start()
        
        # Visual feedback for blink (shows for multiple frames)
        if self.blink_detected_frame > 0:
            # Large visual indicator
            cv2.putText(frame, "BLINK DETECTED!", (frame.shape[1]//2 - 150, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "*CLICK*", (frame.shape[1]//2 - 70, frame.shape[0]//2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Draw blinking indicator circle
            cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2 - 100), 40, (0, 255, 255), 5)
            cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2 - 100), 30, (0, 0, 255), -1)
            
            self.blink_detected_frame -= 1
        
        # Display volume change feedback
        if self.volume_change_feedback:
            current_time = time.time()
            if current_time - self.volume_feedback_time < self.volume_feedback_duration:
                # Large visual feedback in center
                feedback_text = self.volume_change_feedback
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                # Get text size for centering
                text_size = cv2.getTextSize(feedback_text, font, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 150
                
                # Draw background rectangle
                padding = 20
                cv2.rectangle(frame, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                cv2.rectangle(frame, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 255, 255), 2)
                
                # Draw text
                cv2.putText(frame, feedback_text, (text_x, text_y),
                           font, font_scale, (0, 255, 255), thickness)
            else:
                self.volume_change_feedback = None
        
        # Display sensitivity and speed
        mode_text = " | PRECISION" if self.fine_tune_mode else ""
        cv2.putText(frame, f"Sens: {self.sensitivity:.1f}x | Speed: {self.speed_multiplier:.1f}x{mode_text}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Draw GUI buttons
        self.draw_gui_buttons(frame)
        
        # Display features status
        features = []
        if self.focus_lock_enabled:
            features.append("FocusLock")
        if self.blink_enabled:
            features.append("BlinkClick")
        if self.gesture_enabled:
            features.append("Gesture")
        if self.adaptive_dead_zone:
            features.append(f"AdaptDZ")
        
        if features:
            cv2.putText(frame, " | ".join(features), (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Display volume level
        try:
            vol_percent = self.get_volume_percentage()
            is_muted = self.volume.GetMute()
            vol_text = f"Volume: {vol_percent:.0f}%" if not is_muted else "Volume: MUTED"
            vol_color = (0, 255, 0) if not is_muted else (0, 0, 255)
            cv2.putText(frame, vol_text, (frame.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vol_color, 2)
        except:
            pass
        
        return frame
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 70)
        print("♿ ACCESSIBLE EYE TRACKER - 100% HANDS-FREE CONTROL ♿")
        print("🚀 AI Calibration & Eye-Based Gesture Control 🚀")
        print("🌐 Now with Google Integration & Voice Commands! 🌐")
        print("=" * 70)
        print("\n💙 Designed for users with limited mobility - No hands needed!")
        
        print("\n📋 KEYBOARD CONTROLS (Caregiver Setup):")
        print("  R - Start 9-Point Calibration (REQUIRED!)")
        print("  C - Toggle eye control ON/OFF")
        print("  B - Toggle Blink-to-Click ON/OFF")
        print("  Q - Quit")
        
        print("\n🌐 GOOGLE FEATURES (NEW!):")
        print("  G - Open Google Chrome")
        print("  S - Google Search (voice command)")
        print("  T - Google Translate (voice command)")
        print("  V - Voice Command (search, open apps, translate)")
        print("  1 - Open Gmail  |  2 - Open Drive  |  3 - Open Docs")
        print("  4 - Open Sheets |  5 - Open Calendar | 6 - Open Maps")
        print("  7 - Open YouTube | 8 - Open Meet")
        print("\n👁️ ON-SCREEN BUTTONS (EYE CONTROL!):")
        print("  Look at any button and hold gaze for 0.3s to activate")
        print("  H - Toggle on-screen buttons visibility")
        
        print("\n🎯 ACCURACY CONTROLS:")
        print("  ↑/↓ - Increase/Decrease sensitivity")
        print("  +/- - Increase/Decrease speed multiplier")
        print("  P - Toggle PRECISION mode (fine control)")
        print("  F - Toggle Focus Lock (hold gaze to lock cursor)")
        print("  A - Toggle Adaptive Dead Zone")
        
        print("\n🖱️ EYE-BASED CLICK:")
        print("  👁️  Single Blink = Click (Default)")
        print("  🎯 Reliable multi-pattern blink detection")
        
        print("\n🔊 EYE-BASED VOLUME CONTROL:")
        print("  👁️👁️  Quick Double Blink (within 1.2s)  = Volume UP")
        print("  👁️👁️👁️  Quick Triple Blink (within 1.8s) = Volume DOWN")
        print("  👁️━━  Long Blink (hold eyes closed)    = Mute/Unmute")
        print("  ⏱️  Cooldown: 1.5s between volume actions")
        print("  💡 TIP: Blink naturally and quickly for best results")
        
        print("\n🧠 AI-POWERED ACCESSIBILITY FEATURES:")
        print("  ✓ 9-Point Enhanced Calibration System")
        print("  ✓ Polynomial Regression Gaze Mapping Model")
        print("  ✓ Automatic Model Training & Accuracy Scoring")
        print("  ✓ Advanced pupil detection with CLAHE & morphology")
        print("  ✓ Exponential smoothing for responsive tracking")
        print("  ✓ Adaptive dead zone & focus locking")
        print("  ✓ Eye-based blink pattern recognition for gestures")
        print("  ✓ 100% hands-free operation")
        
        print("\n🌐 GOOGLE INTEGRATION (NEW!):")
        print("  🔍 Google Search with voice commands")
        print("  🌍 Google Translate for instant translation")
        print("  📧 Quick access to Gmail, Drive, Docs, Sheets")
        print("  📅 Calendar, Maps, YouTube, Meet integration")
        print("  🎤 Voice recognition using Google Speech API")
        print("  🌐 Chrome browser automation")
        
        print("\n" + "=" * 70)
        print("⚠️  CAREGIVER: Press 'R' to start 9-point calibration first!")
        print("\n👁️  USER CONTROLS (Eyes Only):")
        print("  • Move eyes = Move cursor")
        print("  • Single blink = Click")
        print("  • Double blink = Volume up")
        print("  • Triple blink = Volume down")
        print("  • Long blink = Mute/unmute")
        print("=" * 70 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display control status with mode indicators
            status_parts = []
            status_parts.append("ON" if self.control_enabled else "OFF")
            if self.blink_enabled:
                status_parts.append("BLINK")
            if self.focus_lock_enabled:
                status_parts.append("FL")
            if self.fine_tune_mode:
                status_parts.append("P")
            
            status = "CONTROL: " + " | ".join(status_parts)
            color = (0, 255, 0) if self.control_enabled else (0, 0, 255)
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show frame
            cv2.imshow('Simple Eye Tracker', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                self.control_enabled = not self.control_enabled
                print(f"Control {'ENABLED' if self.control_enabled else 'DISABLED'}")
            elif key == ord('r') or key == ord('R'):
                # Start interactive calibration
                self.start_calibration()
            elif key == 0 or key == 82:  # Up arrow
                self.sensitivity = min(3.0, self.sensitivity + 0.1)
                print(f"Sensitivity increased to {self.sensitivity:.1f}x")
            elif key == 1 or key == 84:  # Down arrow
                self.sensitivity = max(0.5, self.sensitivity - 0.1)
                print(f"Sensitivity decreased to {self.sensitivity:.1f}x")
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(3.0, self.speed_multiplier + 0.1)
                print(f"Speed increased to {self.speed_multiplier:.1f}x")
            elif key == ord('-') or key == ord('_'):
                self.speed_multiplier = max(0.5, self.speed_multiplier - 0.1)
                print(f"Speed decreased to {self.speed_multiplier:.1f}x")
            elif key == ord('['):
                self.dead_zone = max(0, self.dead_zone - 5)
                print(f"Dead zone decreased to {self.dead_zone}px")
            elif key == ord(']'):
                self.dead_zone = min(50, self.dead_zone + 5)
                print(f"Dead zone increased to {self.dead_zone}px")
            elif key == ord('p') or key == ord('P'):
                self.fine_tune_mode = not self.fine_tune_mode
                print(f"Precision mode {'ENABLED' if self.fine_tune_mode else 'DISABLED'}")
            elif key == ord('l') or key == ord('L'):
                self.focus_lock_enabled = not self.focus_lock_enabled
                if not self.focus_lock_enabled:
                    self.focus_locked = False
                    self.focus_lock_start_time = None
                print(f"Focus lock {'ENABLED' if self.focus_lock_enabled else 'DISABLED'}")
            elif key == ord('a') or key == ord('A'):
                self.adaptive_dead_zone = not self.adaptive_dead_zone
                print(f"Adaptive dead zone {'ENABLED' if self.adaptive_dead_zone else 'DISABLED'}")
            elif key == ord('b') or key == ord('B'):
                self.blink_enabled = not self.blink_enabled
                print(f"Blink-to-click {'ENABLED' if self.blink_enabled else 'DISABLED'}")
            # Google Features
            elif key == ord('g') or key == ord('G'):
                print("🌐 Opening Google Chrome...")
                self.open_google_chrome()
            elif key == ord('s') or key == ord('S'):
                print("🔍 Google Search - Speak your query...")
                command = self.listen_for_voice_command()
                if command:
                    self.google_search(command)
            elif key == ord('t') or key == ord('T'):
                print("🌍 Google Translate - Speak the text...")
                command = self.listen_for_voice_command()
                if command:
                    self.google_translate(command)
            elif key == ord('v') or key == ord('V'):
                print("🎤 Voice Command - Speak your command...")
                command = self.listen_for_voice_command()
                if command:
                    self.process_voice_command(command)
            # Quick access to Google services
            elif key == ord('1'):
                self.open_google_service('gmail')
            elif key == ord('2'):
                self.open_google_service('drive')
            elif key == ord('3'):
                self.open_google_service('docs')
            elif key == ord('4'):
                self.open_google_service('sheets')
            elif key == ord('5'):
                self.open_google_service('calendar')
            elif key == ord('6'):
                self.open_google_service('maps')
            elif key == ord('7'):
                self.open_google_service('youtube')
            elif key == ord('h') or key == ord('H'):
                self.show_gui_buttons = not self.show_gui_buttons
                print(f"On-screen GUI buttons: {'VISIBLE' if self.show_gui_buttons else 'HIDDEN'}")
            elif key == ord('8'):
                self.open_google_service('meet')
            # Note: Gesture control (blink patterns for volume) is always enabled for accessibility
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SimpleEyeTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting gracefully...")
        cv2.destroyAllWindows()
