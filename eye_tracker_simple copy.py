
import cv2
import pyautogui
import numpy as np
from collections import deque
import time

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
        self.smooth_buffer = deque(maxlen=5)  # Reduced from 10
        self.gaze_buffer = deque(maxlen=3)  # Reduced from 5
        
        # Blink detection variables
        self.last_blink_time = 0
        self.blink_cooldown = 0.5  # seconds
        self.eyes_detected_history = deque(maxlen=5)
        
        # Control mode
        self.control_enabled = True
        
        # Sensitivity control (1.0 = normal, higher = more sensitive)
        self.sensitivity = 1.5
        self.speed_multiplier = 1.0
        
        # Dead zone to reduce jitter (pixels)
        self.dead_zone = 15
        self.last_cursor_pos = None
        
        # Gaze reference points for calibration
        self.calibration_mode = False
        self.interactive_calibration = True  # Use interactive calibration instead of auto
        self.calibration_complete = False  # Track if interactive calibration is done
        self.calibration_points_targets = [
            (0.1, 0.1),  # Top-left
            (0.9, 0.1),  # Top-right
            (0.5, 0.5),  # Center
            (0.1, 0.9),  # Bottom-left
            (0.9, 0.9),  # Bottom-right
        ]
        self.current_calibration_point = 0
        self.calibration_samples = []
        self.samples_per_point = 30
        self.calibration_delay = 1.5  # seconds to stabilize
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
    
    def preprocess_for_detection(self, gray):
        """Lightweight preprocessing for better detection"""
        # Simple histogram equalization only (fast)
        return cv2.equalizeHist(gray)
    
    def start_calibration(self):
        """Start interactive calibration process"""
        self.calibration_mode = True
        self.calibration_complete = False
        self.current_calibration_point = 0
        self.calibration_samples = []
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
        print("INTERACTIVE CALIBRATION STARTED")
        print("="*60)
        print("Look at each GREEN circle and hold your gaze steady.")
        print("The circle will turn BLUE while collecting data.")
        print("Follow all 5 calibration points.")
        print("="*60 + "\n")
    
    def finish_calibration(self):
        """Complete calibration and compute mapping"""
        if len(self.calibration_gaze_data) >= 5:
            # Build calibration ranges from collected data
            all_x = [g[0] for g in self.calibration_gaze_data]
            all_y = [g[1] for g in self.calibration_gaze_data]
            
            self.calibration_data['min_x'] = min(all_x)
            self.calibration_data['max_x'] = max(all_x)
            self.calibration_data['min_y'] = min(all_y)
            self.calibration_data['max_y'] = max(all_y)
            
            self.frames_processed = 100  # Mark as calibrated
            self.calibration_complete = True  # Mark interactive calibration as done
            
            print("\n" + "="*60)
            print("CALIBRATION COMPLETE!")
            print("="*60)
            print(f"X Range: {self.calibration_data['min_x']:.1f} - {self.calibration_data['max_x']:.1f}")
            print(f"Y Range: {self.calibration_data['min_y']:.1f} - {self.calibration_data['max_y']:.1f}")
            print("Eye tracking is now active - move your eyes!")
            print("Press 'R' to recalibrate if needed.")
            print("="*60 + "\n")
            return True
        return False
        
    def detect_pupil(self, eye_img):
        """Enhanced pupil detection with better accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Multiple threshold approaches for robustness
        _, threshold1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        threshold2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        
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
                
                # Wait for stabilization, then collect
                if current_time - self.calibration_start_time >= self.calibration_delay:
                    self.calibration_samples.append((avg_gaze_x, avg_gaze_y))
                    
                    # Move to next point when enough samples collected
                    if len(self.calibration_samples) >= self.samples_per_point:
                        # Average samples
                        final_x = sum([s[0] for s in self.calibration_samples]) / len(self.calibration_samples)
                        final_y = sum([s[1] for s in self.calibration_samples]) / len(self.calibration_samples)
                        
                        self.calibration_gaze_data.append((final_x, final_y))
                        print(f"Calibration point {self.current_calibration_point + 1} recorded: ({final_x:.1f}, {final_y:.1f})")
                        
                        # Next point
                        self.current_calibration_point += 1
                        self.calibration_samples = []
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
        
        # Weighted averaging (recent positions have more weight)
        weights = np.linspace(0.5, 1.5, len(self.smooth_buffer))
        weights = weights / weights.sum()
        
        avg_x = sum([p[0] * w for p, w in zip(self.smooth_buffer, weights)])
        avg_y = sum([p[1] * w for p, w in zip(self.smooth_buffer, weights)])
        
        # Apply dead zone to reduce jitter
        if self.last_cursor_pos:
            dx = abs(avg_x - self.last_cursor_pos[0])
            dy = abs(avg_y - self.last_cursor_pos[1])
            
            # If movement is within dead zone, keep previous position
            if dx < self.dead_zone and dy < self.dead_zone:
                return self.last_cursor_pos[0], self.last_cursor_pos[1]
        
        smooth_x, smooth_y = int(avg_x), int(avg_y)
        self.last_cursor_pos = (smooth_x, smooth_y)
        
        return smooth_x, smooth_y
    
    def detect_blink(self, num_eyes):
        """Detect blink based on number of eyes detected"""
        self.eyes_detected_history.append(num_eyes)
        
        current_time = time.time()
        
        # Check if eyes disappeared briefly (blink)
        if len(self.eyes_detected_history) >= 5:
            # Pattern: 2 eyes -> 0 eyes -> 2 eyes (blink)
            if (self.eyes_detected_history[0] == 2 and 
                self.eyes_detected_history[1] == 0 and 
                self.eyes_detected_history[2] == 0 and
                self.eyes_detected_history[3] >= 1 and
                current_time - self.last_blink_time > self.blink_cooldown):
                
                self.last_blink_time = current_time
                return True
        
        return False
    
    def update_calibration(self, x, y):
        """Auto-calibrate by tracking min/max positions"""
        self.calibration_data['min_x'] = min(self.calibration_data['min_x'], x)
        self.calibration_data['max_x'] = max(self.calibration_data['max_x'], x)
        self.calibration_data['min_y'] = min(self.calibration_data['min_y'], y)
        self.calibration_data['max_y'] = max(self.calibration_data['max_y'], y)
    
    def map_to_screen(self, gaze_x, gaze_y):
        """Enhanced mapping with sensitivity control and better calibration"""
        if self.frames_processed < 100:
            # Still calibrating
            return None, None
        
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
        """Process frame and track eyes with improved detection"""
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
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Eye region (upper 50% of face)
            eye_region_y = y + int(h * 0.2)
            eye_region_h = int(h * 0.4)
            
            roi_gray = gray[eye_region_y:eye_region_y+eye_region_h, x:x+w]
            roi_color = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w]
            
            # Single eye detection pass - optimized parameters
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(30, 30)
            )
            
            # Take first 2 eyes only
            valid_eyes = sorted(eyes, key=lambda e: e[0])[:2]
            
            for (ex, ey, ew, eh) in valid_eyes:
                num_eyes_detected += 1
                
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
                
                # Control cursor with speed multiplier
                if self.control_enabled:
                    try:
                        pyautogui.moveTo(screen_x, screen_y, duration=0.01 * (1 / self.speed_multiplier))
                    except:
                        pass
                
                # Store as last valid gaze
                self.last_valid_gaze = (screen_x, screen_y)
                
                # Enhanced visual feedback
                cv2.putText(frame, f"Cursor: ({screen_x}, {screen_y})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {int(self.tracking_confidence)}%", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
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
            
            cv2.putText(frame, f"{cal_status} | Eyes: {len(eyes_data)} ({detection_quality})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_color, 2)
            
            if not self.calibration_complete:
                cv2.putText(frame, ">>> Press 'R' to calibrate! <<<", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No eyes detected
            cv2.putText(frame, "NO EYES DETECTED - Adjust lighting/position", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Tips: Face the camera, ensure good lighting", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Detect blinks
        if self.detect_blink(num_eyes_detected):
            if self.control_enabled:
                pyautogui.click()
            cv2.putText(frame, "BLINK - CLICK!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display sensitivity and speed
        cv2.putText(frame, f"Sensitivity: {self.sensitivity:.1f}x | Speed: {self.speed_multiplier:.1f}x", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Display dead zone info
        cv2.putText(frame, f"Dead Zone: {self.dead_zone}px", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 60)
        print("Enhanced Eye Tracker with Interactive Calibration")
        print("="*60)
        print("Controls:")
        print("  R - Start Calibration (PRESS NOW to begin!)")
        print("  C - Toggle control on/off")
        print("  Q - Quit")
        print("  ↑/↓ - Increase/Decrease sensitivity")
        print("  +/- - Increase/Decrease speed")
        print("  [/] - Decrease/Increase dead zone")
        print("\nFeatures:")
        print("  ✓ 5-point interactive calibration")
        print("  ✓ Enhanced pupil detection")
        print("  ✓ Weighted smoothing for stable tracking")
        print("  ✓ Dead zone to reduce jitter")
        print("  ✓ Adjustable sensitivity and speed")
        print("  ✓ Blink to click")
        print("  ✓ Real-time confidence tracking")
        print("="*60)
        print("\n>>> IMPORTANT: Press 'R' to start calibration! <<<\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display control status
            status = "CONTROL: ON" if self.control_enabled else "CONTROL: OFF"
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
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SimpleEyeTracker()
    tracker.run()
