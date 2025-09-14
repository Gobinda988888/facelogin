# PythonAnywhere WSGI Configuration for Face ID Login System

import sys
import os

# Add your project directory to sys.path
project_home = '/home/yourusername/face_login'  # Update with your actual username
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

from app import app as application

# Set environment variables for production
os.environ['SECRET_KEY'] = 'your-super-secret-key-here-change-this'
os.environ['PORT'] = '5000'

if __name__ == "__main__":
    application.run()
