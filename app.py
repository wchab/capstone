from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import base64
import time
from PredictMask import PredictImage
import pandas as pd
import shutil

app = Flask(__name__, static_url_path='/static')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
PRODUCTS_FILE = './static/lipshades.xlsx'
IMAGE_FOLDER = './static/images'

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/virtualtryon/all', methods=['GET', 'POST'])
def virtualtryon():
    product_line_dict = {}
    df = pd.read_excel('lipshades.xlsx')
    colour_dict = dict(zip(df['product_id'], df['color']))
    for folder in os.listdir(f'./static/images'):
        if '.DS_Store' not in folder and folder != 'colours':
            for filename in os.listdir(f'./static/images/{folder}'):
                path = f'{folder}/{filename}'
                if 'png' in filename:
                    product_line_dict[path] = filename.split('.')[0]
    return render_template('virtualtryon.html', image_dict=product_line_dict, colour_dict=colour_dict)

@app.route('/test')
def test():
    return render_template('test.html')

# @app.route('/virtualtryon/intense_volume_matte', methods=['GET', 'POST'])
# def virtualtryon_intense_volume_matte():
#     product_line_dict = {}
#     for filename in os.listdir(f'./static/images/intense_volume_matte'):
#         path = f'intense_volume_matte/{filename}'
#         if 'png' in filename:
#             product_line_dict[path] = filename.split('.')[0]
#     return render_template('virtualtryon.html', image_dict=product_line_dict)

# @app.route('/virtualtryon/reds_of_worth', methods=['GET', 'POST'])
# def virtualtryon_reds_of_worth():
#     product_line_dict = {}
#     for filename in os.listdir(f'./static/images/reds_of_worth'):
#         path = f'reds_of_worth/{filename}'
#         if 'png' in filename:
#             product_line_dict[path] = filename.split('.')[0]
#     return render_template('virtualtryon.html', image_dict=product_line_dict)


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
        df = pd.read_excel(os.path.join('static', PRODUCTS_FILE))

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
        # image_path = os.path.join(IMAGE_FOLDER, image_filename)
        image_path = os.path.join('static', IMAGE_FOLDER, image_filename)

        print(f"Product Name: {product_name}")
        print(f"Product Colour: {product_colour}")
        print(f"Product ID: {product_id}")
        print(f"Image File Name: {image_filename}")
        print(f"Image Path: {image_path}")

        if os.path.exists(image_path):
            # Read the image and convert it to base64 for passing to the HTML template
            with open(image_path, "rb") as product_image_file:
                product_image_data = base64.b64encode(product_image_file.read()).decode("utf-8")
                print("Image data is available")

            return render_template('shadematched.html', product_name=product_name, product_colour=product_colour, product_id=product_id, product_image_data=product_image_data)
        else:
            print("Image file not found")
            return render_template('shadematched.html', product_name=product_name, product_colour=product_colour, product_id=product_id, product_image_data=None)  # Or handle the case where the image file is not found

    else:
        print("Product information not found")
        return render_template('nomatch.html')  # Or handle the case where product information is not found

@app.route('/nomatch')
def nomatch():
    return render_template('nomatch.html')

if __name__ == '__main__':
    app.run(debug=True)