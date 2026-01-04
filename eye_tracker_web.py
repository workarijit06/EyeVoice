"""
Eye Tracker Web Integration
Runs the eye tracker with a web interface for real-time monitoring and control
Includes TensorFlow for adaptive eye movement learning
"""

import cv2
import base64
import threading
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
from eye_tracker_simple import SimpleEyeTracker

# TensorFlow imports (disabled for performance)
TENSORFLOW_AVAILABLE = False
# Uncomment below to enable ML features (requires tensorflow)
# try:
#     import tensorflow as tf
#     from tensorflow import keras
#     from sklearn.preprocessing import StandardScaler
#     TENSORFLOW_AVAILABLE = True
#     print("✅ TensorFlow loaded successfully")
# except (ImportError, Exception) as e:
#     TENSORFLOW_AVAILABLE = False
#     print(f"⚠️ TensorFlow not available: {str(e)[:100]}...")
#     print("   ML features disabled. Install with: pip install tensorflow scikit-learn")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'eye_tracker_secret_2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
tracker = None
camera = None
is_running = False
processing_thread = None
last_processed_frame = None  # Shared frame between tracking and video
frame_lock = threading.Lock()  # Thread lock for frame access

# Machine Learning Model for Eye Movement Prediction
ml_model = None
scaler = None
training_data = {'eye_positions': [], 'cursor_positions': []}
model_trained = False

stats = {
    'eyes_detected': 0,
    'calibrated': False,
    'tracking_active': False,
    'blink_count': 0,
    'click_count': 0,
    'volume_level': 50,
    'accuracy': 0,
    'uptime': 0,
    'ml_accuracy': 0
}

def initialize_tracker():
    """Initialize the eye tracker"""
    global tracker, camera, ml_model, scaler
    tracker = SimpleEyeTracker()
    
    # Enable control and blink detection by default
    tracker.control_enabled = True
    tracker.blink_enabled = True
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Eye tracker initialized - Control: ON, Blink: ON")
    
    # Initialize ML model for eye movement prediction
    if TENSORFLOW_AVAILABLE:
        try:
            ml_model = create_ml_model()
            scaler = StandardScaler()
            print("✅ ML Model initialized for eye movement learning")
        except Exception as e:
            print(f"⚠️ ML Model initialization failed: {e}")

def create_ml_model():
    """Create TensorFlow model for eye position to cursor mapping"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # 4 inputs: eye_x, eye_y, rel_x, rel_y
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='linear')  # 2 outputs: screen_x, screen_y
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_ml_model():
    """Train the ML model with collected eye tracking data"""
    global ml_model, scaler, training_data, model_trained
    
    if not TENSORFLOW_AVAILABLE or ml_model is None:
        return False
    
    if len(training_data['eye_positions']) < 100:
        print("⚠️ Not enough training data (need at least 100 samples)")
        return False
    
    try:
        # Prepare training data
        X = np.array(training_data['eye_positions'])
        y = np.array(training_data['cursor_positions'])
        
        # Normalize input features
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        print("🧠 Training ML model...")
        history = ml_model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        model_trained = True
        final_loss = history.history['val_loss'][-1]
        stats['ml_accuracy'] = max(0, 100 - final_loss)
        
        print(f"✅ ML Model trained! Validation loss: {final_loss:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ ML training failed: {e}")
        return False

def predict_cursor_position(eye_data):
    """Use ML model to predict cursor position from eye data"""
    global ml_model, scaler, model_trained
    
    if not model_trained or ml_model is None:
        return None
    
    try:
        # Prepare input
        eye_input = np.array([eye_data]).reshape(1, -1)
        eye_input_scaled = scaler.transform(eye_input)
        
        # Predict
        prediction = ml_model.predict(eye_input_scaled, verbose=0)
        return prediction[0]
        
    except Exception as e:
        return None
    
def tracking_loop():
    """Main tracking loop with ML integration"""
    global tracker, camera, stats, training_data, is_running, last_processed_frame, frame_lock
    
    frame_count = 0
    
    try:
        while is_running and camera is not None:
            try:
                success, frame = camera.read()
                if not success:
                    print("⚠️ Camera read failed")
                    time.sleep(0.1)
                    continue
                
                # Flip frame horizontally for natural mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame with tracker
                if tracker:
                    try:
                        # The tracker internally handles mouse movement, clicks, and volume gestures
                        processed_frame = tracker.process_frame(frame.copy())
                        
                        # Store processed frame for video streaming
                        with frame_lock:
                            last_processed_frame = processed_frame.copy()
                        
                        # Collect training data every 10 frames
                        if TENSORFLOW_AVAILABLE and frame_count % 10 == 0:
                            if hasattr(tracker, 'last_cursor_pos') and tracker.last_cursor_pos:
                                # Get eye position data (this would need to be exposed from tracker)
                                # For now we'll use placeholder
                                pass
                        
                        # Update stats
                        stats['tracking_active'] = tracker.control_enabled
                        stats['calibrated'] = tracker.calibration_complete
                        stats['accuracy'] = tracker.calibration_accuracy_score if tracker.calibration_complete else 0
                        stats['eyes_detected'] = tracker.num_eyes_detected if hasattr(tracker, 'num_eyes_detected') else 0
                        
                        frame_count += 1
                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                
                time.sleep(0.03)  # ~30 FPS to match video stream
            except Exception as e:
                print(f"⚠️ Tracking loop error: {e}")
                time.sleep(0.1)
    except Exception as e:
        print(f"❌ Fatal tracking loop error: {e}")
    finally:
        print("🛑 Tracking loop stopped")
    
def generate_frames():
    """Generate video frames for streaming - uses frames from tracking_loop"""
    global tracker, camera, stats, last_processed_frame, frame_lock
    
    try:
        while is_running and camera is not None:
            try:
                # Get the processed frame from tracking_loop
                with frame_lock:
                    if last_processed_frame is not None:
                        processed_frame = last_processed_frame.copy()
                    else:
                        # No frame yet, wait
                        time.sleep(0.01)
                        continue
                
                # Update stats (already updated in tracking_loop, but refresh here)
                if tracker:
                    stats['tracking_active'] = tracker.control_enabled
                    stats['calibrated'] = tracker.calibration_complete
                    stats['accuracy'] = tracker.calibration_accuracy_score if tracker.calibration_complete else 0
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"⚠️ Video generation error: {e}")
                time.sleep(0.1)
    except Exception as e:
        print(f"❌ Fatal video generation error: {e}")
    finally:
        print("🛑 Video feed stopped")

def update_stats_loop():
    """Continuously update statistics"""
    global tracker, stats
    start_time = time.time()
    
    try:
        while is_running:
            try:
                if tracker:
                    # Update volume
                    if tracker.volume_available:
                        try:
                            stats['volume_level'] = int(tracker.get_volume_percentage())
                        except Exception as e:
                            pass
                    
                    # Update uptime
                    stats['uptime'] = int(time.time() - start_time)
                    
                    # Emit stats to all connected clients
                    try:
                        socketio.emit('stats_update', stats)
                    except Exception as e:
                        pass
                
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Stats update error: {e}")
                time.sleep(1)
    except Exception as e:
        print(f"❌ Fatal stats loop error: {e}")
    finally:
        print("🛑 Stats loop stopped")
        
        time.sleep(0.5)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('eye_tracker_ui.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_tracking():
    """Start eye tracking"""
    global is_running, tracker, camera, processing_thread
    
    print("📡 Received start tracking request")
    
    if not is_running:
        print("🔧 Initializing tracker...")
        if tracker is None:
            initialize_tracker()
        
        print("✅ Starting tracking threads...")
        is_running = True
        stats['tracking_active'] = True
        
        # Start processing thread for tracking
        processing_thread = threading.Thread(target=tracking_loop, daemon=True)
        processing_thread.start()
        
        # Start stats update thread
        threading.Thread(target=update_stats_loop, daemon=True).start()
        
        print("🎥 Tracking started successfully!")
        return jsonify({'success': True, 'message': 'Tracking started'})
    
    print("⚠️ Already running")
    return jsonify({'success': False, 'message': 'Already running'})

@app.route('/api/stop', methods=['POST'])
def stop_tracking():
    """Stop eye tracking"""
    global is_running, camera
    
    is_running = False
    stats['tracking_active'] = False
    
    if camera:
        camera.release()
        camera = None
    
    return jsonify({'success': True, 'message': 'Tracking stopped'})

@app.route('/api/pause', methods=['POST'])
def pause_tracking():
    """Pause/resume tracking"""
    global tracker
    
    if tracker:
        tracker.control_enabled = not tracker.control_enabled
        status = 'resumed' if tracker.control_enabled else 'paused'
        return jsonify({'success': True, 'message': f'Tracking {status}'})
    return jsonify({'success': False, 'message': 'Tracker not initialized'})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Start calibration"""
    global tracker
    
    if tracker:
        tracker.start_calibration()
        return jsonify({'success': True, 'message': 'Calibration started'})
    return jsonify({'success': False, 'message': 'Tracker not initialized'})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update tracker settings"""
    global tracker
    
    if tracker:
        data = request.json
        
        if 'sensitivity' in data:
            tracker.sensitivity = float(data['sensitivity'])
        
        if 'speed' in data:
            tracker.speed_multiplier = float(data['speed'])
        
        if 'blink_enabled' in data:
            tracker.blink_enabled = bool(data['blink_enabled'])
        
        if 'volume_enabled' in data:
            tracker.gesture_enabled = bool(data['volume_enabled'])
        
        return jsonify({'success': True, 'message': 'Settings updated'})
    return jsonify({'success': False, 'message': 'Tracker not initialized'})

@app.route('/api/precision_mode', methods=['POST'])
def toggle_precision():
    """Toggle precision mode"""
    global tracker
    
    if tracker:
        tracker.fine_tune_mode = not tracker.fine_tune_mode
        status = 'enabled' if tracker.fine_tune_mode else 'disabled'
        return jsonify({'success': True, 'message': f'Precision mode {status}'})
    return jsonify({'success': False, 'message': 'Tracker not initialized'})

@app.route('/api/focus_lock', methods=['POST'])
def toggle_focus_lock():
    """Toggle focus lock"""
    global tracker
    
    if tracker:
        tracker.focus_lock_enabled = not tracker.focus_lock_enabled
        status = 'enabled' if tracker.focus_lock_enabled else 'disabled'
        return jsonify({'success': True, 'message': f'Focus lock {status}'})
    return jsonify({'success': False, 'message': 'Tracker not initialized'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current statistics"""
    return jsonify(stats)

@app.route('/api/train_ml', methods=['POST'])
def train_ml():
    """Train the ML model with collected data"""
    global training_data
    
    if not TENSORFLOW_AVAILABLE:
        return jsonify({'success': False, 'message': 'TensorFlow not available'})
    
    success = train_ml_model()
    if success:
        return jsonify({
            'success': True, 
            'message': f'Model trained with {len(training_data["eye_positions"])} samples',
            'accuracy': stats['ml_accuracy']
        })
    else:
        return jsonify({'success': False, 'message': 'Training failed - need more data'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('stats_update', stats)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("=" * 60)
    print("👁️  Eye Tracker Web Interface")
    print("=" * 60)
    print("Starting server...")
    print(f"📱 Open your browser and navigate to:")
    print(f"   http://localhost:5000")
    print(f"   or http://127.0.0.1:5000")
    print("=" * 60)
    
    # Run the Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
