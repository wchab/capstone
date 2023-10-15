from flask import Flask, render_template, request, redirect, url_for
import os
import base64
import time
from PredictMask import PredictImage
import pandas as pd
import shutil

app = Flask(__name__)
# app = Flask(__name__, static_url_path='/static')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
PRODUCTS_FILE = 'lipshades.xlsx'
IMAGE_FOLDER = 'product_pictures'

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/upload_file', methods=['GET', 'POST'])
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

@app.route('/shadematched')
def shadematched():
    return render_template('shadematched.html')

def get_product_info(product_hexcode):
    try:
        df = pd.read_excel(PRODUCTS_FILE)

        product_name = df[df['hexcode'] == product_hexcode]['name'].iloc[0]
        product_colour = df[df['hexcode'] == product_hexcode]['color'].iloc[0]
        product_id = df[df['hexcode'] == product_hexcode]['product_id'].iloc[0]

        return product_name, product_colour, product_id
    
    except Exception as e:
        print(f"Error loading or searching in the Excel file: {e}")
        return None, None, None

# def get_product_colour(product_hexcode):
#     try:
#         df = pd.read_excel(PRODUCTS_FILE)

#         result = df[df['hexcode'] == product_hexcode]['color'].iloc[0]

#         return result
#     except Exception as e:
#         print(f"Error loading or searching in the Excel file: {e}")
#         return None
    
# def get_product_id(product_hexcode):
#     try:
#         df = pd.read_excel(PRODUCTS_FILE)

#         result = df[df['hexcode'] == product_hexcode]['product_id'].iloc[0]

#         return result
#     except Exception as e:
#         print(f"Error loading or searching in the Excel file: {e}")
#         return None  


@app.route('/shadematched/9F5656')
def match_product():
    product_hexcode = "9F5656"

    product_name, product_colour, product_id = get_product_info(product_hexcode)
    # product_name = get_product_name(product_hexcode)
    # product_colour = get_product_colour(product_hexcode)
    # product_id = get_product_id(product_hexcode)

    # return render_template('shadematched.html', product_name=product_name, product_colour=product_colour, product_id=product_id)

    if product_id is not None:
        # Assuming the product ID is the name of the image file
        image_filename = f"{product_id}.png"  # Adjust the file extension as needed
        image_path = os.path.join(IMAGE_FOLDER, image_filename)

        if os.path.exists(image_path):
            # Read the image and convert it to base64 for passing to the HTML template
            with open(image_path, "rb") as product_image_file:
                product_image_data = base64.b64encode(product_image_file.read()).decode("utf-8")

            return render_template('shadematched.html', product_name=product_name, product_colour=product_colour, product_id=product_id, product_image_data=product_image_data)
        else:
            return render_template('nomatch.html')  # Or handle the case where the image file is not found

    else:
        return render_template('nomatch.html')  # Or handle the case where product information is not found

@app.route('/nomatch')
def nomatch():
    return render_template('nomatch.html')

if __name__ == '__main__':
    app.run(debug=True)