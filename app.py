from flask import Flask, render_template, request, redirect, url_for
import os
import base64
import time
from PredictMask import PredictImage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = os.path.join(file.filename)
        file.save(filename)
        PredictImage().predict_mask(file.filename)
        # Read the image and convert it to base64 for passing to the HTML template
        
        with open(filename, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
        return render_template('index.html', image_data=image_data)
    else:
        return "Invalid file format. Allowed file formats are: png, jpg, jpeg, gif"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return f'<h1>Uploaded Image:</h1><img src="{url_for("static", filename="uploads/" + filename)}">'

if __name__ == '__main__':
    app.run(debug=True)