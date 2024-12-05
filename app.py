from flask import Flask, request, render_template, redirect, url_for, session
from model_loader import model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'Nonbangkok'

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            previous_file = session.get('last_uploaded_file')
            if previous_file and os.path.exists(previous_file):
                os.remove(previous_file)

            file.save(filepath)
            session['last_uploaded_file'] = filepath

            try:
                image = preprocess_image(filepath)
                prediction = model.predict(image)
                prediction_class = 'Diseased' if prediction[0][0] < 0.5 else 'Good'

                return render_template('result.html', filename=filename, prediction=prediction_class)
            except Exception as e:
                os.remove(filepath)
                session.pop('last_uploaded_file', None)
                return f"An error occurred: {str(e)}", 500

    previous_file = session.get('last_uploaded_file')
    if previous_file and os.path.exists(previous_file):
        os.remove(previous_file)
        session.pop('last_uploaded_file', None)

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)