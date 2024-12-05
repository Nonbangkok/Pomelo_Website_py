# app.py
from flask import Flask, request, render_template, redirect, url_for
from model_loader import model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Adjust target_size as per your model's input shape
    image = load_img(image_path, target_size=(256, 256))  # Example size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # Normalize or preprocess as per your model requirements
    image /= 255.0
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':
        # Check if a file is part of the request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            # Save the file to the upload folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Preprocess the image and make prediction
                image = preprocess_image(filepath)
                prediction = model.predict(image)
                # Assuming binary classification and sigmoid activation
                prediction_class = 'Diseased' if prediction[0][0] < 0.5 else 'Good'

                # Remove the image after prediction
                os.remove(filepath)

                return render_template('result.html', filename=filename, prediction=prediction_class)
            except Exception as e:
                # Handle errors and clean up the file if needed
                os.remove(filepath)
                return f"An error occurred: {str(e)}", 500

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render uses the PORT env variable
    app.run(host='0.0.0.0', port=port)