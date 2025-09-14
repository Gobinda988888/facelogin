#!/bin/bash

# Face ID Login System - PythonAnywhere Deployment Script
# Run this script in your PythonAnywhere bash console

echo "🚀 Face ID Login System - PythonAnywhere Setup"
echo "============================================="

# Navigate to project directory
cd ~/face_login

# Create and activate virtual environment
echo "📦 Setting up virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# If OpenCV fails, try headless version
if [ $? -ne 0 ]; then
    echo "⚠️ Standard OpenCV failed, trying headless version..."
    pip install opencv-python-headless
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p known_faces
mkdir -p static/js
mkdir -p static/css
mkdir -p templates

# Set proper permissions
echo "🔐 Setting permissions..."
chmod -R 755 .
chmod -R 755 known_faces

# Test imports
echo "🧪 Testing imports..."
python3 -c "import cv2, numpy, flask, PIL; print('✅ All imports successful!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Configure your WSGI file in the Web tab"
    echo "2. Set the virtual environment path: $(pwd)/venv"
    echo "3. Update SECRET_KEY in your .env file"
    echo "4. Click 'Reload' in the Web tab"
    echo "5. Visit your app: https://yourusername.pythonanywhere.com"
    echo ""
    echo "🎉 Your Face ID system is ready for deployment!"
else
    echo "❌ Setup failed. Check the error messages above."
fi
