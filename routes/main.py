from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
import os

# Initialize Blueprint
main_bp = Blueprint('main', __name__)

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)


@main_bp.route('/')
def home():
    return render_template('index.html')


@main_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join('uploads/', file.filename)
        file.save(file_path)
        return redirect(url_for('main.display_image', filename=file.filename))


@main_bp.route('/display/<filename>')
def display_image(filename):
    return render_template('solution.html', filename=filename)


# Serve uploaded files properly
@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads/', filename)
