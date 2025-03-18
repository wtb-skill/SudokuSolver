# app.py
from flask import Flask
from routes.main import main_bp  # Import routes from the main.py file


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Register blueprint for route handling
app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(debug=True)

