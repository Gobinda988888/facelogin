from flask import Flask, render_template, request, jsonify, session
import cv2
import numpy as np
import os
from PIL import Image
import io
import base64
import pickle
import secrets
from html import escape
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
def landing():
    return redirect(url_for('portfolio'))
    '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face ID / Portfolio</title>
        <style>
            :root { --ink: #081014; --paper: #edf1e8; --lime: #c8f169; --orange: #ff744d; --muted: #9daaa4; --line: rgba(237,241,232,.14); }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            html { scroll-behavior: smooth; }
            body { background: var(--ink); color: var(--paper); font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; overflow-x: hidden; opacity: 0; animation: pageIn 1s .1s forwards; }
            body::after { content: ''; position: fixed; inset: 0; z-index: 20; pointer-events: none; background: var(--ink); animation: curtainUp 1.1s cubic-bezier(.76,0,.24,1) forwards; }
            @keyframes pageIn { to { opacity: 1; } }
            @keyframes curtainUp { 0% { transform: translateY(0); } 100% { transform: translateY(-100%); } }
            body::before { content: ''; position: fixed; inset: 0; pointer-events: none; opacity: .16; background-image: linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px); background-size: 52px 52px; mask-image: linear-gradient(to bottom, black, transparent 78%); }
            .shell { width: min(1180px, calc(100% - 40px)); margin: auto; position: relative; }
            nav { display: flex; justify-content: space-between; align-items: center; padding: 1.5rem 0; border-bottom: 1px solid var(--line); }
            .brand { font-weight: 800; letter-spacing: 2px; font-size: .78rem; }
            .brand i { display: inline-block; width: 9px; height: 9px; background: var(--lime); border-radius: 50%; margin-right: 8px; box-shadow: 0 0 16px var(--lime); }
            nav a { color: var(--muted); font-size: .78rem; text-decoration: none; margin-left: 1.4rem; }
            nav a:hover { color: var(--lime); }
            .hero { min-height: 78vh; display: grid; grid-template-columns: 1.1fr .9fr; align-items: center; gap: 4rem; padding: 5rem 0; }
            .hero-media { position: absolute; inset: 0; z-index: -1; overflow: hidden; opacity: .25; pointer-events: none; }
            .hero-media video { width: 100%; height: 100%; object-fit: cover; filter: saturate(.65) contrast(1.1); }
            .hero-media::after { content: ''; position: absolute; inset: 0; background: linear-gradient(90deg, var(--ink) 5%, rgba(8,16,20,.78) 48%, rgba(8,16,20,.35)), linear-gradient(0deg, var(--ink), transparent 45%, var(--ink)); }
            .developer-tag { display: inline-flex; align-items: center; gap: .5rem; color: var(--muted); border: 1px solid var(--line); padding: .45rem .7rem; border-radius: 999px; font: .7rem Consolas, monospace; margin-bottom: 1.2rem; animation: blink 2.4s ease-in-out infinite; }
            .developer-tag b { color: var(--lime); font-weight: 400; }
            @keyframes blink { 50% { border-color: rgba(200,241,105,.55); box-shadow: 0 0 22px rgba(200,241,105,.08); } }
            .kicker { color: var(--lime); text-transform: uppercase; letter-spacing: 3px; font-size: .7rem; margin-bottom: 1.4rem; }
            h1 { font-size: clamp(3.8rem, 9vw, 8.8rem); line-height: .86; letter-spacing: -6px; max-width: 760px; }
            h1 em { color: var(--lime); font-style: normal; }
            .copy { color: var(--muted); max-width: 490px; margin: 2rem 0; font-size: 1.04rem; }
            .actions { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
            .primary { background: var(--lime); color: var(--ink); padding: .9rem 1.2rem; border-radius: 999px; font-weight: 800; font-size: .78rem; text-decoration: none; transition: transform .25s, box-shadow .25s; }
            .primary:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(200,241,105,.24); }
            .hint { color: var(--muted); font-size: .72rem; }
            .visual { aspect-ratio: 1; max-width: 390px; margin-left: auto; border: 1px solid var(--line); border-radius: 50%; display: grid; place-items: center; position: relative; background: radial-gradient(circle, rgba(200,241,105,.18), transparent 47%), #101e1d; box-shadow: 0 0 100px rgba(200,241,105,.08); animation: float 5s ease-in-out infinite; transition: transform .2s ease-out; overflow: hidden; }
            .profile-photo { width: 82%; height: 82%; object-fit: cover; border-radius: 50%; border: 4px solid rgba(237,241,232,.75); box-shadow: 0 0 35px rgba(200,241,105,.35); position: relative; z-index: 1; filter: saturate(.9) contrast(1.05); }
            .visual.has-photo::before, .visual.has-photo::after { z-index: 2; pointer-events: none; }
            .visual::before, .visual::after { content: ''; position: absolute; inset: 13%; border: 1px solid rgba(200,241,105,.25); border-radius: 50%; animation: orbit 12s linear infinite; }
            .visual::after { inset: 28%; border-color: rgba(255,116,77,.4); animation-direction: reverse; animation-duration: 8s; }
            .core { width: 30%; aspect-ratio: 1; border-radius: 50%; background: var(--lime); box-shadow: 0 0 50px var(--lime); animation: corePulse 2.4s ease-in-out infinite; }
            .visual-label { position: absolute; right: -12%; top: 28%; color: var(--lime); font-size: .65rem; letter-spacing: 2px; }
            @keyframes float { 50% { transform: translateY(-12px) rotate(3deg); } }
            @keyframes orbit { to { transform: rotate(360deg); } }
            @keyframes corePulse { 50% { transform: scale(1.12); box-shadow: 0 0 75px var(--lime); } }
            .preview { border-top: 1px solid var(--line); padding: 4rem 0 6rem; }
            .preview-head { display: flex; justify-content: space-between; align-items: end; margin-bottom: 1.5rem; }
            .preview h2 { font-size: clamp(2rem, 5vw, 4rem); line-height: .95; letter-spacing: -2px; }
            .preview-head p { color: var(--muted); max-width: 280px; font-size: .8rem; }
            .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
            .card { min-height: 230px; padding: 1.3rem; border: 1px solid var(--line); background: #10191b; display: flex; flex-direction: column; justify-content: space-between; transition: transform .3s, border-color .3s; }
            .card { animation: cardIn .8s both; transform-style: preserve-3d; }
            .card:nth-child(2) { animation-delay: .12s; }
            .card:nth-child(3) { animation-delay: .24s; }
            .card:hover { transform: translateY(-6px) rotate(-1deg); border-color: var(--lime); }
            @keyframes cardIn { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }
            .art { height: 82px; background: repeating-linear-gradient(135deg, transparent 0 14px, rgba(200,241,105,.2) 15px 16px); }
            .card:nth-child(2) .art { background: radial-gradient(circle at 70% 30%, var(--orange), transparent 11%), repeating-radial-gradient(circle, transparent 0 17px, rgba(255,116,77,.25) 18px 19px); }
            .card:nth-child(3) .art { background: linear-gradient(120deg, transparent 45%, rgba(141,243,255,.35) 46% 54%, transparent 55%), #172128; }
            .card.image-card { padding: 0; overflow: hidden; }
            .card.image-card .art { height: 100%; min-height: 230px; margin: 0; background: none; }
            .card.image-card img { width: 100%; height: 100%; object-fit: cover; filter: saturate(.8) contrast(1.05); transition: transform .6s, filter .6s; }
            .card.image-card:hover img { transform: scale(1.08); filter: saturate(1.05) contrast(1.08); }
            .card.image-card .image-caption { position: absolute; align-self: flex-start; margin: 1.3rem; padding: .45rem .7rem; background: rgba(8,16,20,.75); border: 1px solid rgba(237,241,232,.2); border-radius: 999px; color: var(--lime); font-size: .68rem; letter-spacing: 1px; }
            .tag { color: var(--muted); font-size: .68rem; letter-spacing: 1px; text-transform: uppercase; }
            .card h3 { margin-top: .8rem; font-size: 1.25rem; }
            footer { border-top: 1px solid var(--line); padding: 2rem 0; color: var(--muted); font-size: .72rem; display: flex; justify-content: space-between; }
            .scroll-line { position: fixed; z-index: 5; top: 0; left: 0; height: 3px; width: 0; background: var(--lime); box-shadow: 0 0 14px var(--lime); }
            @media (max-width: 700px) { .shell { width: min(100% - 28px, 1180px); } nav a { display: none; } .hero { grid-template-columns: 1fr; gap: 2.5rem; padding: 4rem 0; } h1 { letter-spacing: -3px; } .visual { width: 72vw; margin: auto; } .cards { grid-template-columns: 1fr; } footer { display: block; } footer span { display: block; margin-top: .5rem; } }
        </style>
    </head>
    <body>
        <div class="scroll-line" id="scrollLine"></div><main class="shell">
            <nav><a class="brand" href="/"><i></i> GOBINDA KUMAR SAHANI</a><div><a href="#work">Work</a><a href="#about">About</a><a href="https://github.com/Gobinda988888" target="_blank" rel="noopener">GitHub</a><a href="/login">Face login</a></div></nav>
            <section class="hero" id="about"><div class="hero-media"><video autoplay muted loop playsinline poster="/static/profile.jpg"><source src="/static/hero-video.mp4" type="video/mp4"></video></div><div><div class="developer-tag"><b>~/gobinda-kumar-sahani</b> developer_mode: true</div><div class="kicker">Developer / Designer / Builder</div><h1>Ideas with a <em>pulse.</em></h1><p class="copy">I am Gobinda Kumar Sahani, a developer who turns sharp ideas into useful, memorable digital experiences. Explore the preview, then unlock the full portfolio with Face ID.</p><div class="actions"><a class="primary" href="/login">Enter with Face ID</a><a class="hint" href="https://github.com/Gobinda988888" target="_blank" rel="noopener">View GitHub ↗</a></div></div><div class="visual has-photo"><img class="profile-photo" src="/static/profile.jpg" alt="Gobinda Kumar Sahani" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"><div class="core" style="display:none"></div><span class="visual-label">BUILD / 01</span></div></section>
            <section class="preview" id="work"><div class="preview-head"><h2>Selected<br>signals.</h2><p>A small preview of the work inside. The complete collection opens after verification.</p></div><div class="cards"><article class="card"><div class="art"></div><div><span class="tag">01 / Product</span><h3>Northstar OS</h3></div></article><article class="card"><div class="art"></div><div><span class="tag">02 / Identity</span><h3>Afterglow</h3></div></article><article class="card image-card"><span class="image-caption">03 / CREATIVE DIRECTION</span><div class="art"><img src="/static/design-image.jpg" alt="Gobinda Kumar Sahani creative work"></div></article></div></section>
            <footer><span>Built by Gobinda Kumar Sahani.</span><span><a href="https://github.com/Gobinda988888" target="_blank" rel="noopener">GitHub / Gobinda988888 ↗</a> · Face ID protected / 2026</span></footer>
        </main>
        <script>
            const scrollLine = document.getElementById('scrollLine');
            const visual = document.querySelector('.visual');
            window.addEventListener('scroll', () => { const max = document.documentElement.scrollHeight - innerHeight; scrollLine.style.width = `${(scrollY / max) * 100}%`; }, { passive: true });
            visual.addEventListener('pointermove', (event) => { const rect = visual.getBoundingClientRect(); const x = (event.clientX - rect.left) / rect.width - .5; const y = (event.clientY - rect.top) / rect.height - .5; visual.style.transform = `perspective(800px) rotateY(${x * 10}deg) rotateX(${y * -10}deg) translateY(-8px)`; });
            visual.addEventListener('pointerleave', () => { visual.style.transform = ''; });
        </script>
    </body>
    </html>
    '''

@app.route('/login')
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
            :root {
                --ink: #07121b;
                --panel: rgba(10, 25, 34, 0.78);
                --line: rgba(157, 239, 255, 0.18);
                --cyan: #8df3ff;
                --blue: #55a7ff;
                --muted: #91a8b5;
            }
            body {
                font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
                background: var(--ink);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                color: white;
                overflow-x: hidden;
            }
            body::before {
                content: '';
                position: fixed;
                inset: 0;
                pointer-events: none;
                background: radial-gradient(circle at 12% 15%, rgba(52, 175, 255, .18), transparent 28%), radial-gradient(circle at 90% 82%, rgba(42, 255, 207, .11), transparent 25%), linear-gradient(120deg, transparent 0 48%, rgba(255,255,255,.025) 48% 49%, transparent 49%);
                background-size: auto, auto, 7px 7px;
            }
            .container {
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 3rem 1.25rem;
                text-align: center;
                position: relative;
                animation: rise .8s ease both;
            }
            @keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
            .title {
                font-size: clamp(2.5rem, 7vw, 5.8rem);
                letter-spacing: -2px;
                margin-bottom: .4rem;
                font-weight: 700;
                line-height: .95;
                text-shadow: 0 0 30px rgba(141, 243, 255, .25);
            }
            .subtitle {
                color: var(--muted);
                font-size: .9rem;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 2rem;
            }
            .eyebrow {
                color: var(--cyan);
                font-size: .72rem;
                letter-spacing: 3px;
                text-transform: uppercase;
                margin-bottom: .8rem;
            }
            .camera-container {
                position: relative;
                width: min(86vw, 470px);
                aspect-ratio: 4 / 3;
                border: 1px solid var(--line);
                border-radius: 22px;
                overflow: hidden;
                margin-bottom: 1rem;
                background: linear-gradient(145deg, rgba(40, 93, 112, .25), rgba(2, 11, 17, .9));
                box-shadow: 0 25px 80px rgba(0, 0, 0, .4), inset 0 0 0 1px rgba(255,255,255,.04);
            }
            .camera-container::before, .camera-container::after {
                content: '';
                position: absolute;
                z-index: 2;
                width: 42px;
                height: 42px;
                border: 2px solid var(--cyan);
                opacity: .8;
            }
            .camera-container::before { top: 18px; left: 18px; border-right: 0; border-bottom: 0; }
            .camera-container::after { right: 18px; bottom: 18px; border-left: 0; border-top: 0; }
            .scan-line {
                position: absolute;
                z-index: 3;
                left: 8%;
                right: 8%;
                height: 2px;
                top: 20%;
                background: var(--cyan);
                box-shadow: 0 0 18px 4px rgba(141,243,255,.65);
                opacity: 0;
            }
            .camera-container.active .scan-line { opacity: 1; animation: scan 2.8s ease-in-out infinite; }
            @keyframes scan { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(220px); } }
            #video {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: none;
                filter: saturate(.85) contrast(1.08);
            }
            #canvas {
                display: none;
            }
            .camera-placeholder {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: .65rem;
                font-size: .9rem;
                color: var(--muted);
            }
            .camera-placeholder::before {
                content: '◉';
                color: var(--cyan);
                font-size: 2.2rem;
                animation: pulse 1.8s ease-in-out infinite;
            }
            @keyframes pulse { 50% { opacity: .35; transform: scale(.86); } }
            .status-row {
                width: min(86vw, 470px);
                display: flex;
                justify-content: space-between;
                color: var(--muted);
                font-size: .72rem;
                letter-spacing: 1px;
                text-transform: uppercase;
                margin-bottom: 1.4rem;
            }
            .status-row span:first-child::before { content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #61727b; margin-right: 7px; }
            .camera-container.active + .status-row span:first-child::before { background: #4dffc1; box-shadow: 0 0 10px #4dffc1; }
            .mode-switch {
                display: grid;
                grid-template-columns: 1fr 1fr;
                width: min(86vw, 470px);
                padding: 4px;
                gap: 4px;
                border: 1px solid var(--line);
                border-radius: 12px;
                background: rgba(255,255,255,.04);
                margin-bottom: 1rem;
            }
            .buttons {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                width: min(86vw, 470px);
            }
            .btn {
                padding: .95rem 1.2rem;
                border: none;
                border-radius: 11px;
                font-size: .86rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform .25s, box-shadow .25s, background .25s;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }
            .btn-primary {
                background: linear-gradient(100deg, #70e7ff, #4e96ff);
                box-shadow: 0 12px 28px rgba(63, 165, 255, .2);
                color: white;
            }
            .btn-primary:hover:not(:disabled) {
                transform: translateY(-3px);
                box-shadow: 0 16px 32px rgba(63, 165, 255, .35);
            }
            .btn-secondary {
                background: rgba(255,255,255,.04);
                color: var(--muted);
                border: 1px solid var(--line);
            }
            .mode-switch .btn { border: 0; padding: .7rem; }
            .mode-switch .btn.active, .btn-secondary:hover:not(:disabled) {
                color: white;
                background: rgba(141, 243, 255, .13);
                border-color: rgba(141,243,255,.4);
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
                padding: 1rem 1.1rem;
                border: 1px solid var(--line);
                border-radius: 11px;
                font-size: 1rem;
                background: rgba(255, 255, 255, 0.06);
                color: white;
                outline: none;
                transition: border .2s, box-shadow .2s;
            }
            .input-group input:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(141,243,255,.1); }
            .message {
                width: min(86vw, 470px);
                margin-top: 1rem;
                padding: .9rem 1rem;
                border-radius: 10px;
                font-size: .86rem;
                font-weight: 500;
            }
            .input-group input::placeholder { color: #718791; }
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
            .loading-overlay {
                position: fixed;
                inset: 0;
                z-index: 10;
                display: grid;
                place-items: center;
                background: rgba(3, 11, 16, .76);
                backdrop-filter: blur(12px);
                opacity: 0;
                pointer-events: none;
                transition: opacity .25s;
            }
            .loading-overlay.visible { opacity: 1; pointer-events: auto; }
            .loading-card { width: min(88vw, 340px); padding: 2rem; border: 1px solid var(--line); border-radius: 18px; background: var(--panel); text-align: left; box-shadow: 0 24px 70px rgba(0,0,0,.45); }
            .loader-ring { width: 48px; height: 48px; border: 2px solid rgba(141,243,255,.18); border-top-color: var(--cyan); border-right-color: var(--blue); border-radius: 50%; animation: spin .8s linear infinite; margin-bottom: 1.2rem; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .loading-card h2 { font-size: 1.15rem; margin-bottom: .4rem; }
            .loading-card p { color: var(--muted); font-size: .82rem; margin-bottom: 1.2rem; }
            .loading-steps { display: grid; gap: .55rem; color: #607780; font-size: .75rem; }
            .loading-steps span::before { content: '○'; display: inline-block; width: 20px; color: #607780; }
            .loading-steps span.active { color: var(--cyan); }
            .loading-steps span.active::before { content: '●'; color: var(--cyan); }
            @media (max-width: 768px) {
                .container { padding: 1rem; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="eyebrow">Secure identity gateway / 01</div>
            <h1 class="title">Face ID</h1>
            <p class="subtitle">Unlock with a look. It's that simple.</p>

            <div class="camera-container" id="cameraFrame">
                <video id="video" autoplay muted playsinline></video>
                <canvas id="canvas"></canvas>
                <div class="scan-line"></div>
                <div class="camera-placeholder" id="placeholder">
                    <span>Camera is standing by</span>
                </div>
            </div>
            <div class="status-row"><span id="cameraStatus">Camera offline</span><span>Encrypted session</span></div>

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
                
                <div class="mode-switch">
                    <button class="btn btn-secondary active" id="loginMode" onclick="showLogin()">Login Mode</button>
                    <button class="btn btn-secondary" id="registerMode" onclick="showRegister()">Register Mode</button>
                </div>
            </div>

            <div id="message"></div>
        </div>
        <div class="loading-overlay" id="loadingOverlay" aria-live="polite">
            <div class="loading-card">
                <div class="loader-ring"></div>
                <h2 id="loadingTitle">Reading your face</h2>
                <p id="loadingText">Matching your encrypted face signature...</p>
                <div class="loading-steps">
                    <span class="active" id="stepCapture">Capture image</span>
                    <span id="stepDetect">Detect face landmarks</span>
                    <span id="stepVerify">Verify identity</span>
                </div>
            </div>
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

            function setLoading(isLoading, mode = 'login') {
                const overlay = document.getElementById('loadingOverlay');
                document.getElementById('loadingTitle').textContent = mode === 'register' ? 'Creating your Face ID' : 'Reading your face';
                document.getElementById('loadingText').textContent = mode === 'register' ? 'Building your encrypted face signature...' : 'Matching your encrypted face signature...';
                overlay.classList.toggle('visible', isLoading);
                if (isLoading) {
                    document.querySelectorAll('.loading-steps span').forEach(step => step.classList.remove('active'));
                    document.getElementById('stepCapture').classList.add('active');
                    setTimeout(() => document.getElementById('stepDetect').classList.add('active'), 450);
                    setTimeout(() => document.getElementById('stepVerify').classList.add('active'), 1000);
                }
            }

            function showLogin() {
                currentMode = 'login';
                document.getElementById('loginSection').classList.remove('hidden');
                document.getElementById('registerSection').classList.add('hidden');
                document.getElementById('loginMode').classList.add('active');
                document.getElementById('registerMode').classList.remove('active');
            }

            function showRegister() {
                currentMode = 'register';
                document.getElementById('loginSection').classList.add('hidden');
                document.getElementById('registerSection').classList.remove('hidden');
                document.getElementById('loginMode').classList.remove('active');
                document.getElementById('registerMode').classList.add('active');
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
                            document.getElementById('cameraFrame').classList.add('active');
                            document.getElementById('cameraStatus').textContent = 'Camera online';
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
                    setLoading(true, 'login');
                    
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
                        setTimeout(() => window.location.href = '/portfolio', 1200);
                    } else {
                        showMessage(result.message, 'error');
                    }
                } catch (error) {
                    showMessage('Login failed. Please try again.', 'error');
                } finally {
                    setLoading(false);
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
                    setLoading(true, 'register');
                    
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
                    setLoading(false);
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

@app.route('/portfolio')
def portfolio():
    is_authenticated = bool(session.get('user'))
    user_name = escape(str(session.get('user', 'Public visitor')))
    login_time = datetime.now().strftime('%b %d, %Y / %I:%M %p')
    access_label = 'Private portfolio / Face verified' if is_authenticated else 'Public portfolio / Welcome'
    access_copy = 'A private portfolio for a curious maker who turns sharp ideas into useful, memorable digital experiences.' if is_authenticated else 'A public portfolio for a curious maker who turns sharp ideas into useful, memorable digital experiences.'
    access_action = '<button class="logout" onclick="logout()">Sign out</button>' if is_authenticated else '<a class="logout" href="/login">Face login</a>'
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{user_name} / Portfolio</title>
        <style>
            :root {{
                --ink: #081014;
                --paper: #edf1e8;
                --lime: #c8f169;
                --orange: #ff744d;
                --muted: #9daaa4;
                --line: rgba(237,241,232,.14);
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            html {{ scroll-behavior: smooth; }}
            body {{ background: var(--ink); color: var(--paper); font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; line-height: 1.5; overflow-x: hidden; opacity: 0; animation: pageIn 1s .1s forwards; }}
            body::after {{ content: ''; position: fixed; inset: 0; z-index: 20; pointer-events: none; background: var(--ink); animation: curtainUp 1.1s cubic-bezier(.76,0,.24,1) forwards; }}
            @keyframes pageIn {{ to {{ opacity: 1; }} }}
            @keyframes curtainUp {{ 0% {{ transform: translateY(0); }} 100% {{ transform: translateY(-100%); }} }}
            body::before {{ content: ''; position: fixed; inset: 0; pointer-events: none; opacity: .17; background-image: linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px); background-size: 52px 52px; mask-image: linear-gradient(to bottom, black, transparent 75%); }}
            a {{ color: inherit; text-decoration: none; }}
            .shell {{ width: min(1180px, calc(100% - 40px)); margin: auto; position: relative; }}
            nav {{ display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 0; border-bottom: 1px solid var(--line); }}
            .brand {{ font-weight: 800; letter-spacing: 2px; font-size: .78rem; }}
            .brand i {{ display: inline-block; width: 9px; height: 9px; background: var(--lime); border-radius: 50%; margin-right: 8px; box-shadow: 0 0 16px var(--lime); }}
            .nav-links {{ display: flex; gap: 1.5rem; color: var(--muted); font-size: .78rem; }}
            .nav-links a:hover {{ color: var(--lime); }}
            .logout {{ border: 1px solid var(--line); background: transparent; color: var(--paper); border-radius: 999px; padding: .55rem .9rem; cursor: pointer; font: inherit; font-size: .72rem; }}
            .logout:hover {{ border-color: var(--lime); color: var(--lime); }}
            .hero {{ min-height: 72vh; display: grid; grid-template-columns: 1.15fr .85fr; align-items: center; gap: 4rem; padding: 5rem 0 4rem; }}
            .hero-media {{ position: absolute; inset: 0; z-index: -1; overflow: hidden; opacity: .2; pointer-events: none; }}
            .hero-media video {{ width: 100%; height: 100%; object-fit: cover; filter: saturate(.65) contrast(1.1); }}
            .hero-media::after {{ content: ''; position: absolute; inset: 0; background: linear-gradient(90deg, var(--ink) 5%, rgba(8,16,20,.78) 48%, rgba(8,16,20,.35)), linear-gradient(0deg, var(--ink), transparent 45%, var(--ink)); }}
            .kicker {{ color: var(--lime); text-transform: uppercase; letter-spacing: 3px; font-size: .7rem; margin-bottom: 1.4rem; }}
            .developer-tag {{ display: inline-flex; gap: .5rem; color: var(--muted); border: 1px solid var(--line); padding: .45rem .7rem; border-radius: 999px; font: .7rem Consolas, monospace; margin-bottom: 1.2rem; animation: blink 2.4s ease-in-out infinite; }}
            .developer-tag b {{ color: var(--lime); font-weight: 400; }}
            @keyframes blink {{ 50% {{ border-color: rgba(200,241,105,.55); box-shadow: 0 0 22px rgba(200,241,105,.08); }} }}
            h1 {{ font-size: clamp(3.6rem, 9vw, 8.5rem); line-height: .86; letter-spacing: -6px; max-width: 750px; }}
            h1 em {{ color: var(--lime); font-style: normal; }}
            .hero-copy {{ color: var(--muted); max-width: 480px; margin-top: 2rem; font-size: 1.05rem; }}
            .hero-meta {{ display: flex; gap: 2rem; margin-top: 2.2rem; color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: 1px; }}
            .hero-meta strong {{ display: block; color: var(--paper); font-size: 1rem; margin-bottom: .2rem; }}
            .orbit {{ aspect-ratio: 1; max-width: 390px; margin-left: auto; border: 1px solid var(--line); border-radius: 50%; display: grid; place-items: center; position: relative; background: radial-gradient(circle, rgba(200,241,105,.18), transparent 48%), #101e1d; box-shadow: 0 0 100px rgba(200,241,105,.08); animation: float 5s ease-in-out infinite; transition: transform .2s ease-out; }}
            .profile-photo {{ width: 82%; height: 82%; object-fit: cover; border-radius: 50%; border: 4px solid rgba(237,241,232,.75); box-shadow: 0 0 35px rgba(200,241,105,.35); position: relative; z-index: 1; filter: saturate(.9) contrast(1.05); }}
            .orbit::before, .orbit::after {{ content: ''; position: absolute; inset: 13%; border: 1px solid rgba(200,241,105,.25); border-radius: 50%; animation: orbitSpin 12s linear infinite; }}
            .orbit::after {{ inset: 28%; border-color: rgba(255,116,77,.4); animation-direction: reverse; animation-duration: 8s; }}
            .orbit-core {{ width: 30%; aspect-ratio: 1; border-radius: 50%; background: var(--lime); box-shadow: 0 0 50px var(--lime); position: relative; animation: corePulse 2.4s ease-in-out infinite; }}
            .orbit-core::after {{ content: 'FACE VERIFIED'; position: absolute; left: 130%; top: 25%; color: var(--lime); font-size: .65rem; line-height: 1.5; letter-spacing: 2px; width: 120px; }}
            @keyframes float {{ 50% {{ transform: translateY(-12px) rotate(3deg); }} }}
            @keyframes orbitSpin {{ to {{ transform: rotate(360deg); }} }}
            @keyframes corePulse {{ 50% {{ transform: scale(1.12); box-shadow: 0 0 75px var(--lime); }} }}
            .section {{ padding: 5rem 0; border-top: 1px solid var(--line); }}
            .section-head {{ display: flex; justify-content: space-between; align-items: end; margin-bottom: 2rem; }}
            .section-head h2 {{ font-size: clamp(2rem, 5vw, 4rem); line-height: .95; letter-spacing: -2px; }}
            .section-head p {{ color: var(--muted); max-width: 300px; font-size: .82rem; }}
            .work-grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 1rem; }}
            .work-card {{ min-height: 300px; padding: 1.5rem; border: 1px solid var(--line); background: #10191b; display: flex; flex-direction: column; justify-content: space-between; transition: transform .3s, border-color .3s; }}
            .work-card {{ animation: cardIn .8s both; }}
            .work-card:nth-child(2) {{ animation-delay: .12s; }}
            .work-card:nth-child(3) {{ animation-delay: .24s; }}
            .work-card:nth-child(4) {{ animation-delay: .36s; }}
            .work-card:hover {{ transform: translateY(-7px) rotate(-1deg); border-color: var(--lime); }}
            @keyframes cardIn {{ from {{ opacity: 0; transform: translateY(22px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            .work-card:nth-child(1) {{ grid-column: span 7; background: linear-gradient(135deg, #1e3329, #10191b 65%); }}
            .work-card:nth-child(2) {{ grid-column: span 5; background: linear-gradient(135deg, #33231e, #10191b 65%); }}
            .work-card:nth-child(3) {{ grid-column: span 5; }}
            .work-card:nth-child(4) {{ grid-column: span 7; background: linear-gradient(135deg, #202637, #10191b 65%); }}
            .card-top {{ display: flex; justify-content: space-between; color: var(--muted); font-size: .7rem; letter-spacing: 1px; }}
            .card-art {{ height: 95px; margin: 1rem 0; border: 1px solid rgba(255,255,255,.14); background: repeating-linear-gradient(135deg, transparent 0 14px, rgba(200,241,105,.18) 15px 16px); }}
            .project-image {{ background: linear-gradient(rgba(8,16,20,.05), rgba(8,16,20,.2)), url('/static/design-image.jpg') center / cover; filter: saturate(.85); }}
            .work-card:nth-child(2) .card-art {{ background: radial-gradient(circle at 70% 30%, var(--orange), transparent 10%), repeating-radial-gradient(circle, transparent 0 17px, rgba(255,116,77,.25) 18px 19px); }}
            .card-title {{ font-size: 1.5rem; margin-bottom: .3rem; }}
            .card-desc {{ color: var(--muted); font-size: .78rem; }}
            .about-grid {{ display: grid; grid-template-columns: .8fr 1.2fr; gap: 4rem; }}
            .about-grid h3 {{ font-size: 2rem; max-width: 360px; line-height: 1.05; }}
            .about-grid p {{ color: var(--muted); max-width: 590px; margin-bottom: 1rem; }}
            .chips {{ display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.5rem; }}
            .chip {{ border: 1px solid var(--line); padding: .45rem .7rem; border-radius: 999px; color: var(--lime); font-size: .7rem; }}
            footer {{ border-top: 1px solid var(--line); padding: 2rem 0 3rem; display: flex; justify-content: space-between; color: var(--muted); font-size: .72rem; }}
            .scroll-line {{ position: fixed; z-index: 5; top: 0; left: 0; height: 3px; width: 0; background: var(--lime); box-shadow: 0 0 14px var(--lime); }}
            .reveal {{ opacity: 0; transform: translateY(18px); transition: opacity .7s, transform .7s; }}
            .reveal.show {{ opacity: 1; transform: translateY(0); }}
            @media (max-width: 700px) {{ .shell {{ width: min(100% - 28px, 1180px); }} .nav-links {{ display: none; }} .hero {{ grid-template-columns: 1fr; gap: 2.5rem; padding: 4rem 0 3rem; }} h1 {{ letter-spacing: -3px; }} .orbit {{ width: 70vw; margin: auto; }} .work-card:nth-child(n) {{ grid-column: span 12; min-height: 260px; }} .about-grid {{ grid-template-columns: 1fr; gap: 1.5rem; }} footer {{ display: block; }} footer span {{ display: block; margin-top: .5rem; }} }}
        </style>
    </head>
    <body>
        <div class="scroll-line" id="scrollLine"></div><main class="shell">
            <nav>
                <a class="brand" href="#top"><i></i> {user_name.upper()} / PORTFOLIO</a>
                <div class="nav-links"><a href="#work">Work</a><a href="#about">About</a><a href="#contact">Contact</a></div>
                {access_action}
            </nav>
            <section class="hero" id="top">
                <div class="hero-media"><video autoplay muted loop playsinline poster="/static/profile.jpg"><source src="/static/hero-video.mp4" type="video/mp4"></video></div>
                <div class="reveal"><div class="developer-tag"><b>~/gobinda-kumar-sahani</b> developer_mode: true</div><div class="kicker">{access_label}</div><h1>Ideas with a <em>pulse.</em></h1><p class="hero-copy">{access_copy}</p><div class="hero-meta"><div><strong>{user_name}</strong>{'Signed in user' if is_authenticated else 'Visiting now'}</div><div><strong>{login_time}</strong>Current session</div></div></div>
                <div class="orbit reveal"><img class="profile-photo" src="/static/profile.jpg" alt="Gobinda Kumar Sahani" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"><div class="orbit-core" style="display:none"></div></div>
            </section>
            <section class="section reveal" id="work"><div class="section-head"><h2>Selected<br>signals.</h2><p>A few things I have been shaping across design, code and the spaces between them.</p></div><div class="work-grid">
                <article class="work-card"><div class="card-top"><span>01 / PRODUCT</span><span>2026</span></div><div class="card-art"></div><div><h3 class="card-title">Northstar OS</h3><p class="card-desc">A calmer command center for teams moving fast.</p></div></article>
                <article class="work-card"><div class="card-top"><span>02 / IDENTITY</span><span>2025</span></div><div class="card-art"></div><div><h3 class="card-title">Afterglow</h3><p class="card-desc">A visual language for late-night creators.</p></div></article>
                <article class="work-card"><div class="card-top"><span>03 / WEB</span><span>2025</span></div><div class="card-art"></div><div><h3 class="card-title">Field Notes</h3><p class="card-desc">Editorial tools for people who notice details.</p></div></article>
                <article class="work-card"><div class="card-top"><span>04 / CREATIVE</span><span>2026</span></div><div class="card-art project-image"></div><div><h3 class="card-title">Human / Machine</h3><p class="card-desc">Exploring warmer interfaces for intelligent systems.</p></div></article>
            </div></section>
            <section class="section about-grid reveal" id="about"><h3>Good work should feel obvious in hindsight.</h3><div><p>I am Gobinda Kumar Sahani, a developer who builds digital products with a bias toward clarity, character and momentum.</p><p>This portfolio is unlocked with face verification, so the work starts with a little trust.</p><div class="chips"><span class="chip">Python / Flask</span><span class="chip">Frontend craft</span><span class="chip">Motion systems</span><span class="chip">Face ID</span></div></div></section>
            <footer id="contact"><span>Available for thoughtful collaborations.</span><span><a href="https://github.com/Gobinda988888" target="_blank" rel="noopener">GitHub / Gobinda988888 ↗</a> · Built with intent / {datetime.now().year}</span></footer>
        </main>
        <script>
            const observer = new IntersectionObserver((entries) => entries.forEach(entry => {{ if (entry.isIntersecting) entry.target.classList.add('show'); }}), {{ threshold: .12 }});
            document.querySelectorAll('.reveal').forEach(item => observer.observe(item));
            const scrollLine = document.getElementById('scrollLine');
            const orbit = document.querySelector('.orbit');
            window.addEventListener('scroll', () => {{ const max = document.documentElement.scrollHeight - innerHeight; scrollLine.style.width = `${{(scrollY / max) * 100}}%`; }}, {{ passive: true }});
            orbit.addEventListener('pointermove', (event) => {{ const rect = orbit.getBoundingClientRect(); const x = (event.clientX - rect.left) / rect.width - .5; const y = (event.clientY - rect.top) / rect.height - .5; orbit.style.transform = `perspective(800px) rotateY(${{x * 10}}deg) rotateX(${{y * -10}}deg) translateY(-8px)`; }});
            orbit.addEventListener('pointerleave', () => {{ orbit.style.transform = ''; }});
            async function logout() {{ try {{ await fetch('/logout', {{ method: 'POST' }}); }} finally {{ window.location.href = '/'; }} }}
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