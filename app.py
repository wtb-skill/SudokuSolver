# app.py
from flask import Flask
from flask_session import Session
from routes.sudoku_solver import sudoku_bp
from routes.dev_tools import dev_tools_bp
import shutil
import os

# Delete old session files on every restart
session_folder = './flask_session_data'
shutil.rmtree(session_folder, ignore_errors=True)
os.makedirs(session_folder, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for the session
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on disk
app.config['SESSION_FILE_DIR'] = './flask_session_data'
app.config['SESSION_PERMANENT'] = False  # Only lasts for the session unless overridden

# Initialize the session extension
Session(app)

# Register blueprint for route handling
app.register_blueprint(sudoku_bp)
app.register_blueprint(dev_tools_bp)

if __name__ == '__main__':
    app.run(debug=True)

