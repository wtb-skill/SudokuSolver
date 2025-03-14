from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
from modules.image_processing import create_warped_image
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
        # Save the uploaded file to the 'uploads' folder
        file_path = os.path.join('uploads/', file.filename)
        file.save(file_path)

        # Call the new helper function to process and save the warped image
        warped_image_path = create_warped_image(file_path)

        # Pass both the uploaded image filename and warped image filename to the template
        return render_template('solution.html', filename=file.filename,
                               warped_filename=warped_image_path.split('/')[-1])


# Serve uploaded files properly from the 'uploads' directory
@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
