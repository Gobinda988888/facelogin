from flask import Flask, render_template, request, jsonify, session
import cv2
import numpy as np
import os
from PIL import Image
import io
import base64
import pickle
import secrets
from functools import wraps
from flask import redirect, url_for
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

app = Flask(__name__)
# Use environment variable for secret key or generate a secure one
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Create a directory to store known faces
FACE_STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'known_faces')
if not os.path.exists(FACE_STORAGE_DIR):
    os.makedirs(FACE_STORAGE_DIR)

# Dictionary to store known face encodings
known_face_encodings = {}
known_face_names = {}

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_known_faces():
    """Load known faces from the known_faces directory"""
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    
    try:
        for filename in os.listdir(FACE_STORAGE_DIR):
            if filename.endswith('.pkl'):
                name = filename[:-4]  # Remove .pkl extension
                features_path = os.path.join(FACE_STORAGE_DIR, filename)
                try:
                    with open(features_path, 'rb') as f:
                        known_face_encodings[name] = pickle.load(f)
                    known_face_names[name] = name
                    print(f"Loaded face encoding for: {name}")
                except Exception as e:
                    print(f"Error loading face encoding for {name}: {e}")
    except Exception as e:
        print(f"Error accessing face storage directory: {e}")
        # Create directory if it doesn't exist
        if not os.path.exists(FACE_STORAGE_DIR):
            os.makedirs(FACE_STORAGE_DIR)

def extract_face_features(face_image):
    """Extract face features using multiple methods for better comparison"""
    # Convert to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
    
    # Resize to standard size
    resized = cv2.resize(gray, (200, 200))
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(resized)
    
    # Extract features using different methods
    features = {}
    
    # 1. Raw pixel values (normalized)
    features['pixels'] = equalized.flatten() / 255.0
    
    # 2. Local Binary Pattern (LBP) for texture
    try:
        # Simple LBP implementation
        lbp = np.zeros_like(equalized)
        for i in range(1, equalized.shape[0] - 1):
            for j in range(1, equalized.shape[1] - 1):
                center = equalized[i, j]
                binary_string = ''
                for di in [-1, -1, -1, 0, 0, 1, 1, 1]:
                    for dj in [-1, 0, 1, -1, 1, -1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        neighbor = equalized[i + di, j + dj]
                        binary_string += '1' if neighbor > center else '0'
                        break
                lbp[i, j] = int(binary_string[:8], 2) if len(binary_string) >= 8 else 0
        
        # Get LBP histogram
        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        features['lbp'] = hist_lbp.flatten() / np.sum(hist_lbp)
    except:
        features['lbp'] = np.zeros(256)
    
    # 3. Histogram of gradients
    try:
        # Calculate gradients
        grad_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get histogram of gradients
        hist_grad = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [256], [0, 256])
        features['gradients'] = hist_grad.flatten() / np.sum(hist_grad)
    except:
        features['gradients'] = np.zeros(256)
    
    return features

def compare_face_features(features1, features2):

    """Compares two sets of face features."""
    try:
        # Calculate similarities for different features
        similarities = []
        
        # Pixel similarity (using correlation)
        corr_pixels = np.corrcoef(features1['pixels'], features2['pixels'])[0, 1]
        if not np.isnan(corr_pixels):
            similarities.append(max(0, corr_pixels))
        
        # LBP similarity (using correlation)
        corr_lbp = np.corrcoef(features1['lbp'], features2['lbp'])[0, 1]
        if not np.isnan(corr_lbp):
            similarities.append(max(0, corr_lbp))
        
        # Gradient similarity (using correlation)
        corr_grad = np.corrcoef(features1['gradients'], features2['gradients'])[0, 1]
        if not np.isnan(corr_grad):
            similarities.append(max(0, corr_grad))
        
        # Calculate final similarity as weighted average
        if similarities:
            final_similarity = np.mean(similarities)
        else:
            final_similarity = 0.0
        
        print(f"Face comparison similarities: {similarities}, Final: {final_similarity}")
        
        # Adjusted threshold for better matching
        is_match = final_similarity > float(os.environ.get('FACE_SIMILARITY_THRESHOLD', 0.35))
        return is_match, final_similarity
        
    except Exception as e:
        print(f"Error comparing features: {e}")
        return False, 0.0

# Decorator to require login for certain routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>Face ID Login</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                color: white;
            }
            .container {
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem;
                text-align: center;
            }
            .title {
                font-size: 3rem;
                margin-bottom: 0.5rem;
                font-weight: 300;
            }
            .subtitle {
                font-size: 1.2rem;
                margin-bottom: 3rem;
                opacity: 0.9;
            }
            .camera-container {
                position: relative;
                width: 320px;
                height: 320px;
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                overflow: hidden;
                margin-bottom: 2rem;
                background: rgba(0, 0, 0, 0.2);
            }
            #video {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: none;
            }
            #canvas {
                display: none;
            }
            .camera-placeholder {
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
                color: rgba(255, 255, 255, 0.7);
            }
            .buttons {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                width: 100%;
                max-width: 320px;
            }
            .btn {
                padding: 1rem 2rem;
                border: none;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .btn-primary {
                background: #4facfe;
                color: white;
            }
            .btn-primary:hover:not(:disabled) {
                background: #00c9ff;
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3);
            }
            .btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .input-group {
                margin-bottom: 1rem;
            }
            .input-group input {
                width: 100%;
                padding: 1rem;
                border: none;
                border-radius: 25px;
                font-size: 1rem;
                background: rgba(255, 255, 255, 0.1);
                color: white;
            }
            .input-group input::placeholder {
                color: rgba(255, 255, 255, 0.7);
            }
            .message {
                margin-top: 1rem;
                padding: 1rem;
                border-radius: 10px;
                font-weight: 500;
            }
            .message.success {
                background: rgba(46, 204, 113, 0.2);
                border: 1px solid #2ecc71;
                color: #2ecc71;
            }
            .message.error {
                background: rgba(231, 76, 60, 0.2);
                border: 1px solid #e74c3c;
                color: #e74c3c;
            }
            .hidden {
                display: none !important;
            }
            @media (max-width: 768px) {
                .title { font-size: 2rem; }
                .camera-container { width: 280px; height: 280px; }
                .container { padding: 1rem; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">Face ID</h1>
            <p class="subtitle">Unlock with a look. It's that simple.</p>

            <div class="camera-container">
                <video id="video" autoplay muted playsinline></video>
                <canvas id="canvas"></canvas>
                <div class="camera-placeholder" id="placeholder">
                    Click "Start Camera" to begin
                </div>
            </div>

            <div class="buttons">
                <button class="btn btn-secondary" id="startCamera" onclick="startCamera()">
                    Start Camera
                </button>
                
                <div id="loginSection" class="hidden">
                    <button class="btn btn-primary" id="loginBtn" onclick="loginWithFace()">
                        Unlock
                    </button>
                </div>
                
                <div id="registerSection" class="hidden">
                    <div class="input-group">
                        <input type="text" id="registerName" placeholder="Enter your name" maxlength="50">
                    </div>
                    <button class="btn btn-primary" id="registerBtn" onclick="registerFace()">
                        Set Up Face ID
                    </button>
                </div>
                
                <button class="btn btn-secondary" onclick="showLogin()">Login Mode</button>
                <button class="btn btn-secondary" onclick="showRegister()">Register Mode</button>
            </div>

            <div id="message"></div>
        </div>

        <script>
            let video, canvas, ctx;
            let currentMode = 'login';
            let stream = null;

            function showMessage(text, type) {
                const messageDiv = document.getElementById('message');
                messageDiv.textContent = text;
                messageDiv.className = `message ${type}`;
                messageDiv.classList.remove('hidden');
                setTimeout(() => messageDiv.classList.add('hidden'), 5000);
            }

            function showLogin() {
                currentMode = 'login';
                document.getElementById('loginSection').classList.remove('hidden');
                document.getElementById('registerSection').classList.add('hidden');
            }

            function showRegister() {
                currentMode = 'register';
                document.getElementById('loginSection').classList.add('hidden');
                document.getElementById('registerSection').classList.remove('hidden');
            }

            async function startCamera() {
                try {
                    video = document.getElementById('video');
                    canvas = document.getElementById('canvas');
                    ctx = canvas.getContext('2d');
                    
                    console.log('Starting camera...');
                    console.log('Browser info:', navigator.userAgent);
                    console.log('Location:', location.href);
                    console.log('Protocol:', location.protocol);
                    
                    // Check if getUserMedia is supported with better detection
                    const hasModernAPI = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
                    const hasLegacyAPI = !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia);
                    
                    console.log('API availability:', { hasModernAPI, hasLegacyAPI });
                    
                    if (!hasModernAPI && !hasLegacyAPI) {
                        throw new Error('No camera API available. Please use Chrome, Firefox, or Safari with HTTPS.');
                    }
                    
                    // Set up legacy APIs if modern API not available
                    if (!hasModernAPI) {
                        navigator.getUserMedia = navigator.getUserMedia || 
                                               navigator.webkitGetUserMedia || 
                                               navigator.mozGetUserMedia || 
                                               navigator.msGetUserMedia;
                    }
                    
                    // Multiple constraint attempts for better Android compatibility
                    const constraintAttempts = [
                        // First attempt: High quality with front camera
                        {
                            video: {
                                width: { ideal: 640, min: 320, max: 1280 },
                                height: { ideal: 480, min: 240, max: 720 },
                                facingMode: { ideal: 'user' },
                                frameRate: { ideal: 30, min: 15, max: 30 }
                            },
                            audio: false
                        },
                        // Second attempt: Lower quality
                        {
                            video: {
                                width: { ideal: 480, min: 320 },
                                height: { ideal: 360, min: 240 },
                                facingMode: 'user'
                            },
                            audio: false
                        },
                        // Third attempt: Basic constraints
                        {
                            video: {
                                facingMode: 'user'
                            },
                            audio: false
                        },
                        // Fourth attempt: Any video
                        {
                            video: true,
                            audio: false
                        }
                    ];

                    let success = false;
                    let lastError = null;

                    for (let i = 0; i < constraintAttempts.length; i++) {
                        try {
                            console.log(`Trying camera constraint ${i + 1}:`, constraintAttempts[i]);
                            
                            if (hasModernAPI) {
                                stream = await navigator.mediaDevices.getUserMedia(constraintAttempts[i]);
                            } else {
                                // Fallback for older browsers
                                stream = await new Promise((resolve, reject) => {
                                    navigator.getUserMedia(constraintAttempts[i], resolve, reject);
                                });
                            }
                            
                            success = true;
                            console.log(`Camera started successfully with constraint ${i + 1}`);
                            break;
                        } catch (error) {
                            console.log(`Constraint ${i + 1} failed:`, error);
                            lastError = error;
                            continue;
                        }
                    }

                    if (!success) {
                        throw lastError || new Error('All camera constraints failed');
                    }

                    video.srcObject = stream;
                    
                    // Wait for video to be ready
                    video.onloadedmetadata = () => {
                        video.play().then(() => {
                            document.getElementById('placeholder').style.display = 'none';
                            video.style.display = 'block';
                            document.getElementById('startCamera').textContent = 'Camera Active';
                            document.getElementById('startCamera').disabled = true;
                            showLogin();
                            showMessage('Camera started successfully!', 'success');
                        }).catch(err => {
                            console.error('Video play error:', err);
                            showMessage('Camera started but video play failed. Try refreshing the page.', 'error');
                        });
                    };

                    video.onerror = (err) => {
                        console.error('Video error:', err);
                        showMessage('Video stream error. Please try again.', 'error');
                    };
                    
                } catch (error) {
                    console.error('Camera error:', error);
                    let errorMessage = 'Camera access failed: ';
                    
                    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                        errorMessage += 'Please allow camera permissions. Click the camera icon in your browser address bar and allow access.';
                        
                        // Add specific Chrome instructions
                        if (/Chrome/i.test(navigator.userAgent)) {
                            errorMessage += ' In Chrome: Click the camera icon next to the address bar → Allow.';
                        }
                    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                        errorMessage += 'No camera found. Please check if your device has a working camera.';
                    } else if (error.name === 'NotSupportedError' || error.name === 'ConstraintNotSatisfiedError') {
                        errorMessage += 'Camera not supported. Try using Chrome or Firefox with HTTPS.';
                    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                        errorMessage += 'Camera is being used by another application. Please close other camera apps and refresh.';
                    } else if (error.name === 'OverconstrainedError') {
                        errorMessage += 'Camera constraints not supported. Try a different device orientation or refresh.';
                    } else if (error.message.includes('API')) {
                        errorMessage += 'Camera API not available. Please use HTTPS or try a different browser.';
                    } else {
                        errorMessage += `${error.message || 'Unknown error'}. Try refreshing or using HTTPS.`;
                    }
                    
                    // Add protocol-specific suggestions
                    if (location.protocol !== 'https:') {
                        errorMessage += ' HTTPS is recommended for camera access.';
                    }
                    
                    showMessage(errorMessage, 'error');
                }
            }

            function captureImage() {
                if (!video || video.videoWidth === 0) {
                    showMessage('Camera not ready. Please wait and try again.', 'error');
                    return null;
                }
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                return canvas.toDataURL('image/jpeg', 0.8);
            }

            async function loginWithFace() {
                try {
                    document.getElementById('loginBtn').disabled = true;
                    document.getElementById('loginBtn').textContent = 'Processing...';
                    
                    const imageData = captureImage();
                    if (!imageData) return;

                    const response = await fetch('/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    });

                    const result = await response.json();
                    
                    if (result.success) {
                        showMessage(result.message, 'success');
                        setTimeout(() => window.location.href = '/demo', 2000);
                    } else {
                        showMessage(result.message, 'error');
                    }
                } catch (error) {
                    showMessage('Login failed. Please try again.', 'error');
                } finally {
                    document.getElementById('loginBtn').disabled = false;
                    document.getElementById('loginBtn').textContent = 'Unlock';
                }
            }

            async function registerFace() {
                try {
                    const name = document.getElementById('registerName').value.trim();
                    if (!name) {
                        showMessage('Please enter your name', 'error');
                        return;
                    }

                    document.getElementById('registerBtn').disabled = true;
                    document.getElementById('registerBtn').textContent = 'Processing...';
                    
                    const imageData = captureImage();
                    if (!imageData) return;

                    const response = await fetch('/register', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData, name: name })
                    });

                    const result = await response.json();
                    
                    if (result.success) {
                        showMessage(result.message, 'success');
                        document.getElementById('registerName').value = '';
                        setTimeout(() => showLogin(), 2000);
                    } else {
                        showMessage(result.message, 'error');
                    }
                } catch (error) {
                    showMessage('Registration failed. Please try again.', 'error');
                } finally {
                    document.getElementById('registerBtn').disabled = false;
                    document.getElementById('registerBtn').textContent = 'Set Up Face ID';
                }
            }

            // Initialize
            document.addEventListener('DOMContentLoaded', () => {
                showLogin();
                
                // Detect mobile and browser
                const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
                const isAndroid = /Android/i.test(navigator.userAgent);
                const isChrome = /Chrome/i.test(navigator.userAgent);
                const isFirefox = /Firefox/i.test(navigator.userAgent);
                
                console.log('Device info:', { isMobile, isAndroid, isChrome, isFirefox });
                console.log('Navigator mediaDevices:', !!navigator.mediaDevices);
                console.log('Navigator getUserMedia:', !!navigator.getUserMedia);
                console.log('Navigator webkitGetUserMedia:', !!navigator.webkitGetUserMedia);
                console.log('Location protocol:', location.protocol);
                
                // More comprehensive camera API check
                const hasModernAPI = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
                const hasLegacyAPI = !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia);
                const hasAnyAPI = hasModernAPI || hasLegacyAPI;
                
                console.log('Camera API availability:', { hasModernAPI, hasLegacyAPI, hasAnyAPI });
                
                if (!hasAnyAPI) {
                    showMessage('Camera API not available. Please use Chrome, Firefox, or Safari with HTTPS.', 'error');
                    document.getElementById('startCamera').disabled = true;
                    return;
                }
                
                // Show specific instructions for Android
                if (isAndroid) {
                    const instructions = document.createElement('div');
                    instructions.className = 'android-instructions';
                    instructions.style.cssText = `
                        background: rgba(255, 193, 7, 0.2);
                        border: 1px solid #ffc107;
                        color: #ffc107;
                        padding: 1rem;
                        border-radius: 10px;
                        margin-bottom: 1rem;
                        font-size: 0.9rem;
                        line-height: 1.4;
                    `;
                    
                    let instructionText = '';
                    if (!isChrome && !isFirefox) {
                        instructionText = `
                            <strong>Android Tips:</strong><br>
                            • Use Chrome or Firefox browser for best results<br>
                            • Allow camera permissions when prompted<br>
                            • Make sure no other apps are using the camera
                        `;
                    } else {
                        instructionText = `
                            <strong>Android Ready:</strong><br>
                            • Click "Allow" when asked for camera permission<br>
                            • If camera fails, try refreshing the page<br>
                            • Make sure camera is not being used by other apps
                        `;
                    }
                    
                    // Add HTTPS recommendation if not using HTTPS
                    if (location.protocol !== 'https:') {
                        instructionText += '<br>• HTTPS recommended for better camera support';
                    }
                    
                    instructions.innerHTML = instructionText;
                    
                    const container = document.querySelector('.camera-container');
                    container.parentNode.insertBefore(instructions, container);
                }
                
                // Add debug info for troubleshooting
                const debugInfo = document.createElement('div');
                debugInfo.style.cssText = `
                    background: rgba(0, 0, 0, 0.1);
                    padding: 0.5rem;
                    border-radius: 5px;
                    margin-bottom: 1rem;
                    font-size: 0.8rem;
                    opacity: 0.7;
                `;
                debugInfo.innerHTML = `
                    Debug: ${isChrome ? 'Chrome' : isFirefox ? 'Firefox' : 'Other'} | 
                    ${hasModernAPI ? 'Modern API' : hasLegacyAPI ? 'Legacy API' : 'No API'} | 
                    ${location.protocol}
                `;
                
                const container = document.querySelector('.camera-container');
                container.parentNode.insertBefore(debugInfo, container);
                
                // Add permission request button for mobile
                if (isMobile) {
                    const permButton = document.createElement('button');
                    permButton.className = 'btn btn-secondary';
                    permButton.textContent = 'Test Camera Permission';
                    permButton.style.marginBottom = '1rem';
                    permButton.onclick = async () => {
                        try {
                            console.log('Testing camera permission...');
                            let stream;
                            
                            if (hasModernAPI) {
                                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                            } else if (navigator.getUserMedia) {
                                stream = await new Promise((resolve, reject) => {
                                    navigator.getUserMedia({ video: true }, resolve, reject);
                                });
                            } else if (navigator.webkitGetUserMedia) {
                                stream = await new Promise((resolve, reject) => {
                                    navigator.webkitGetUserMedia({ video: true }, resolve, reject);
                                });
                            }
                            
                            if (stream) {
                                stream.getTracks().forEach(track => track.stop());
                                showMessage('Camera permission granted! You can now start the camera.', 'success');
                                permButton.style.display = 'none';
                            }
                        } catch (error) {
                            console.error('Permission test error:', error);
                            showMessage(`Camera test failed: ${error.name} - ${error.message}`, 'error');
                            
                            // Provide specific solutions
                            if (error.name === 'NotAllowedError') {
                                showMessage('Please allow camera permissions in browser settings and refresh.', 'error');
                            } else if (location.protocol !== 'https:') {
                                showMessage('Try accessing via HTTPS for better camera support.', 'error');
                            }
                        }
                    };
                    
                    const startButton = document.getElementById('startCamera');
                    startButton.parentNode.insertBefore(permButton, startButton);
                }
            });

            window.addEventListener('beforeunload', () => {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/register', methods=['POST'])
def register():
    try:
        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        name = request.json['name']
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(image_cv, 1.1, 4)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        
        # Save the image
        image_path = os.path.join(FACE_STORAGE_DIR, f'{name}.jpg')
        cv2.imwrite(image_path, image_cv)
        
        # Extract features for the detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = image_cv[y:y+h, x:x+w]

        # Extract and save features
        features = extract_face_features(face_crop)
        features_path = os.path.join(FACE_STORAGE_DIR, f'{name}.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        
        # Update known faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': 'Face registered successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/login', methods=['POST'])
def login():
    try:
        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(image_cv, 1.1, 4)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        
        # Extract features from the detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = image_cv[y:y+h, x:x+w]
        login_features = extract_face_features(face_crop)
        
        # Compare with known faces
        best_match = None
        best_similarity = 0.0
        
        for name, known_features in known_face_encodings.items():
            is_match, similarity = compare_face_features(known_features, login_features)
            print(f"Comparing with {name}: similarity = {similarity}")
            
            if is_match and similarity > best_similarity:
                best_match = name
                best_similarity = similarity
        
        if best_match:
            session['user'] = best_match
            return jsonify({'success': True, 'message': f'Welcome {best_match}! (Confidence: {best_similarity:.2f})'})
        
        return jsonify({'success': False, 'message': 'Face is not registered or similarity too low'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/demo')
@login_required
def demo():
    user = session.get('user', None)
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Welcome - Face ID</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: white;
                text-align: center;
                padding: 2rem;
            }}
            .welcome-container {{
                background: rgba(255, 255, 255, 0.1);
                padding: 3rem;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                max-width: 500px;
                width: 100%;
            }}
            .welcome-title {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
                font-weight: 300;
            }}
            .welcome-message {{
                font-size: 1.2rem;
                margin-bottom: 2rem;
                opacity: 0.9;
            }}
            .user-info {{
                background: rgba(255, 255, 255, 0.1);
                padding: 1.5rem;
                border-radius: 15px;
                margin-bottom: 2rem;
            }}
            .user-name {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}
            .login-time {{
                font-size: 1rem;
                opacity: 0.8;
            }}
            .features {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            .feature {{
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
            }}
            .feature-icon {{
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }}
            .feature-title {{
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}
            .feature-desc {{
                font-size: 0.9rem;
                opacity: 0.8;
            }}
            .btn {{
                padding: 1rem 2rem;
                border: none;
                border-radius: 50px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin: 0.5rem;
                text-decoration: none;
                display: inline-block;
            }}
            .btn-primary {{
                background: #4facfe;
                color: white;
            }}
            .btn-secondary {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            }}
            @media (max-width: 768px) {{
                .welcome-container {{
                    padding: 2rem;
                }}
                .welcome-title {{
                    font-size: 2rem;
                }}
                .features {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="welcome-container">
            <h1 class="welcome-title">🎉 Welcome!</h1>
            <p class="welcome-message">You have successfully logged in using Face ID</p>
            
            <div class="user-info">
                <div class="user-name">👤 {user}</div>
                <div class="login-time">🕐 Logged in at {__import__('datetime').datetime.now().strftime('%I:%M %p on %B %d, %Y')}</div>
            </div>
            
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">🔒</div>
                    <div class="feature-title">Secure Login</div>
                    <div class="feature-desc">Face recognition technology for secure access</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Fast Access</div>
                    <div class="feature-desc">Quick and seamless authentication</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">📱</div>
                    <div class="feature-title">Mobile Ready</div>
                    <div class="feature-desc">Works perfectly on mobile devices</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">High Accuracy</div>
                    <div class="feature-desc">Advanced face matching algorithms</div>
                </div>
            </div>
            
            <div>
                <button class="btn btn-secondary" onclick="logout()">Logout</button>
                <a href="/" class="btn btn-primary">Back to Login</a>
            </div>
        </div>

        <script>
            async function logout() {{
                try {{
                    const response = await fetch('/logout', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    
                    const result = await response.json();
                    if (result.success) {{
                        window.location.href = '/';
                    }}
                }} catch (error) {{
                    console.error('Logout error:', error);
                    window.location.href = '/';
                }}
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/debug')
@login_required
def debug():
    """Debug route to check registered faces"""
    faces_info = []
    for name, path in known_face_encodings.items():
        faces_info.append({
            'name': name,
            'path': path,
            'exists': os.path.exists(path)
        })
    return jsonify({'registered_faces': faces_info})

@app.route('/test_match', methods=['POST'])
def test_match():
    """Test route to check face matching with detailed output"""
    try:
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(image_cv, 1.1, 4)
        
        results = {
            'faces_detected': len(faces),
            'face_coordinates': faces.tolist() if len(faces) > 0 else [],
            'matches': []
        }
        
        if len(faces) == 0:
            results['message'] = 'No faces detected in the uploaded image'
            return jsonify(results)
        
        # Extract features from the detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = image_cv[y:y+h, x:x+w]
        test_features = extract_face_features(face_crop);
        
        # Test against all known faces
        for name, known_features in known_face_encodings.items():
            is_match, similarity = compare_face_features(known_features, test_features)
            results['matches'].append({
                'name': name,
                'similarity': float(similarity),
                'is_match': is_match,
                'threshold': 0.4
            })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/info')
def info():
    """System information for debugging"""
    import platform
    import socket
    
    # Get local IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "Unable to detect"
    
    return jsonify({
        'system': platform.system(),
        'python_version': platform.python_version(),
        'local_ip': local_ip,
        'registered_faces': len(known_face_encodings),
        'opencv_available': True,
        'message': 'Face ID Login System is running!'
    })

@app.route('/camera-test')
def camera_test():
    """Simple camera test page for debugging Android issues"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Camera Test</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background: #f0f0f0; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            video { width: 100%; max-width: 400px; border: 2px solid #ddd; border-radius: 10px; }
            button { padding: 10px 20px; margin: 10px; border: none; border-radius: 5px; background: #007bff; color: white; cursor: pointer; }
            .info { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .error { background: #ffe7e7; padding: 15px; border-radius: 5px; margin: 10px 0; color: #d00; }
            .success { background: #e7ffe7; padding: 15px; border-radius: 5px; margin: 10px 0; color: #0a0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 Camera Compatibility Test</h1>
            
            <div id="deviceInfo" class="info">
                <strong>Device Information:</strong><br>
                <span id="userAgent"></span><br>
                <span id="browserInfo"></span>
            </div>
            
            <div id="cameraSupport" class="info">
                <strong>Camera API Support:</strong><br>
                <span id="apiInfo"></span>
            </div>
            
            <button onclick="testCamera()">Test Camera</button>
            <button onclick="testBasicCamera()">Test Basic Camera</button>
            <a href="/"><button>Back to Face ID</button></a>
            
            <div id="messages"></div>
            
            <video id="video" autoplay muted playsinline style="display:none;"></video>
        </div>

        <script>
            // Display device info
            document.getElementById('userAgent').textContent = navigator.userAgent;
            
            const isAndroid = /Android/i.test(navigator.userAgent);
            const isChrome = /Chrome/i.test(navigator.userAgent);
            const isFirefox = /Firefox/i.test(navigator.userAgent);
            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            
            document.getElementById('browserInfo').innerHTML = `
                Android: ${isAndroid ? '✅' : '❌'} | 
                Chrome: ${isChrome ? '✅' : '❌'} | 
                Firefox: ${isFirefox ? '✅' : '❌'} | 
                Mobile: ${isMobile ? '✅' : '❌'}
            `;
            
            // Check API support
            const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
            const hasLegacyAPI = !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia);
            
            document.getElementById('apiInfo').innerHTML = `
                Modern API (mediaDevices.getUserMedia): ${hasGetUserMedia ? '✅' : '❌'}<br>
                Legacy API: ${hasLegacyAPI ? '✅' : '❌'}<br>
                HTTPS: ${location.protocol === 'https:' ? '✅' : '❌ (May be required for camera)'}
            `;
            
            function addMessage(text, type = 'info') {
                const div = document.createElement('div');
                div.className = type;
                div.innerHTML = text;
                document.getElementById('messages').appendChild(div);
            }
            
            async function testCamera() {
                try {
                    addMessage('🔍 Testing camera with optimal settings...', 'info');
                    
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            facingMode: 'user'
                        }
                    });
                    
                    const video = document.getElementById('video');
                    video.srcObject = stream;
                    video.style.display = 'block';
                    
                    addMessage('✅ Camera test successful! Face ID should work.', 'success');
                    
                } catch (error) {
                    addMessage(`❌ Camera test failed: ${error.name} - ${error.message}`, 'error');
                    testBasicCamera();
                }
            }
            
            async function testBasicCamera() {
                try {
                    addMessage('🔍 Testing basic camera access...', 'info');
                    
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    
                    const video = document.getElementById('video');
                    video.srcObject = stream;
                    video.style.display = 'block';
                    
                    addMessage('✅ Basic camera works! Face ID might work with reduced quality.', 'success');
                    
                } catch (error) {
                    addMessage(`❌ Basic camera failed: ${error.name} - ${error.message}`, 'error');
                    
                    if (error.name === 'NotAllowedError') {
                        addMessage('💡 Solution: Allow camera permissions in browser settings', 'info');
                    } else if (error.name === 'NotFoundError') {
                        addMessage('💡 Solution: Check if device has a working camera', 'info');
                    } else if (location.protocol !== 'https:') {
                        addMessage('💡 Solution: Try using HTTPS instead of HTTP', 'info');
                    }
                }
            }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Load known faces on startup
    load_known_faces()
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"🚀 Starting Face ID Login Server")
    print(f"📍 Port: {port}")
    print(f"🏠 Access at: http://localhost:{port}")
    print(f"📱 For mobile: http://[YOUR-IP]:{port}")
    print(f"📊 Registered faces: {len(known_face_encodings)}")
    print("=" * 50)
    
    # For PythonAnywhere deployment, use different configuration
    if os.environ.get('PYTHONANYWHERE_DOMAIN'):
        print("🌐 Running on PythonAnywhere")
        # Don't call app.run() when running on PythonAnywhere
        # The WSGI server will handle this
    else:
        app.run(
            host="0.0.0.0", 
            port=port, 
            debug=debug_mode,
            threaded=True
        )

# Load known faces when module is imported (for PythonAnywhere)
load_known_faces()