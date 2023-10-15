from flask import Flask, render_template, request, redirect, url_for
import os
import base64
import time
from PredictMask import PredictImage

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        uploads_filename = os.path.join("uploads", file.filename)
        file.save(uploads_filename)
        processed_filename = PredictImage().predict_mask(uploads_filename)
        # Read the image and convert it to base64 for passing to the HTML template
        
        with open(uploads_filename, "rb") as raw_image_file:
            raw = base64.b64encode(raw_image_file.read()).decode("utf-8")
        with open(processed_filename, "rb") as processed_image_file:
            processed = base64.b64encode(processed_image_file.read()).decode("utf-8")
        return render_template('upload.html', image_data1=raw, image_data2=processed)
    else:
        return "Invalid file format. Allowed file formats are: png, jpg, jpeg, gif"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return f'<h1>Uploaded Image:</h1><img src="{url_for("static", filename="uploads/" + filename)}">'

if __name__ == '__main__':
    app.run(debug=True)