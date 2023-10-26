from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import base64
import time
import pandas as pd
import concurrent.futures
import asyncio
from LipColorizer import LipColorizer
from flask_executor import Executor
from UserHandler import UserHandler
from ShadeRecommender import ShadeRecommender

app = Flask(__name__, static_url_path='/static')
executor = Executor(app)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
PRODUCTS_FILE = 'lips_loreal.xlsx'
IMAGE_FOLDER = 'product_pictures'

#initialise userhandler class as a centralised platform for front, back and user experience
userhandler = UserHandler()

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def redirect_home():
    return redirect('/home')

@app.route('/api/products', methods=['GET'])
def get_products():
    product_line_dict = {}
    df = pd.read_excel('./static/lipshades.xlsx')
    df['hexcode'] = df['hexcode'].map(lambda x: str(x))
    df['wordsearch'] = (df['name'] + df['color'] + df['hexcode']).map(lambda x: x.lower())
    colour_dict = dict(zip(df['product_id'], df['color']))
    search_dict = dict(zip(df['product_id'], df['wordsearch']))
    hexcode_dict = dict(zip(df['product_id'], df['hexcode']))
    product_line_dict = dict(zip(df['product_id'], df['product_line']))
    for folder in os.listdir(f'./static/images'):
        if '.DS_Store' not in folder and folder != 'colours':
            for filename in os.listdir(f'./static/images/{folder}'):
                if 'png' in filename:
                    product_number = filename.split('.')[0]
                    product_line_dict[product_number] = {
                                                    'color': colour_dict[product_number],
                                                    'hexcode': hexcode_dict[product_number],
                                                    'product_line': product_line_dict[product_number],
                                                    'wordsearch': search_dict[product_number]}
    return jsonify(product_line_dict)

@app.route('/home')
def home():
    return render_template('home.html')


def process_image():
    df = pd.read_excel('./static/lipshades.xlsx')
    df['hexcode'] = df['hexcode'].map(lambda x: str(x))
    hexcode_dict = dict(zip(df['product_id'], df['hexcode']))
    lip_colorizer_model_path = os.path.join("static", "shape_predictor_68_face_landmarks.dat") 
    uploads_filename = os.path.join("static", "playground", "upload.png")
    df = pd.read_excel('./static/lipshades.xlsx')
    df['hexcode'] = df['hexcode'].map(lambda x: str(x))
    hexcode_dict = dict(zip(df['product_id'], df['hexcode']))
    lip_colorizer_model_path = os.path.join("static", "shape_predictor_68_face_landmarks.dat") 
    uploads_filename = os.path.join("static", "playground", "upload.png")
    for key in hexcode_dict.keys():
        destination_filename = os.path.join("static", "playground", "modified", f"{key}.png")
        lipcolorizer = LipColorizer(lip_colorizer_model_path, uploads_filename)
        image = lipcolorizer.colorize_lips(f'#{hexcode_dict[key]}')
        lipcolorizer.saveImage(image, destination_filename)
        print(f"{key}.png processed")

@app.route('/virtualtryon', methods=['GET', 'POST'])
def virtualtryon():
    if request.files:
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            uploads_filename = os.path.join("static", "playground", "upload.png")
            file.save(uploads_filename)
            executor.submit(process_image)
            return render_template('virtualtryon.html')
    else:
        return render_template('virtualtryon.html')

@app.route('/upload/virtualtryon', methods=['GET', 'POST'])
def upload_virtualtryon():
    return render_template('upload_virtualtryon.html')

@app.route('/upload/lipshadefinder', methods=['GET', 'POST'])
def upload_lipshadefinder():
    return render_template('upload_lipshadefinder.html')

def get_product_info(product_hexcode):
    try:
        df = pd.read_excel(os.path.join('static', PRODUCTS_FILE))

        product_name = df[df['hex_colour'] == product_hexcode]['name'].iloc[0]
        # product_colour = df[df['hex_colour'] == product_hexcode]['color'].iloc[0]
        product_price = df[df['hex_colour'] == product_hexcode]['price'].iloc[0]
        product_price_str = f"{product_price:.2f}"
        product_id = df[df['hex_colour'] == product_hexcode]['prod_id'].iloc[0]

        return product_name, product_price_str, product_id
    
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


# @app.route('/shadematched/FFC0CB')
@app.route('/shadematched')
def match_product():
    # product_hexcode = "#FFC0CB"

    # product_name, product_price, product_id = get_product_info(product_hexcode)
    # product_name = get_product_name(product_hexcode)
    # product_colour = get_product_colour(product_hexcode)
    # product_id = get_product_id(product_hexcode)

    # return render_template('shadematched.html', product_name=product_name, product_colour=product_colour, product_id=product_id)

    product_info = userhandler.shaderecommender.product_dic
    print(product_info)
    product_name = product_info["name"]
    product_price = product_info["price"]
    product_price_str = f"{product_price:.2f}"
    product_id = product_info["prod_id"]
    product_link = "https://www.google.com/"
    product_hexcode = product_info["hex_colour"][1:]

    if product_id is not None:
        # Assuming the product ID is the name of the image file
        image_filename = f"{product_id}.png"  # Adjust the file extension as needed
        # image_path = os.path.join(IMAGE_FOLDER, image_filename)
        image_path = os.path.join('static', IMAGE_FOLDER, image_filename)

        print(f"Product Name: {product_name}")
        print(f"Product Price: {product_price_str}")
        print(f"Product ID: {product_id}")
        print(f"Product hexcode: {product_hexcode}")
        print(f"Image File Name: {image_filename}")
        print(f"Image Path: {image_path}")

        if os.path.exists(image_path):
            # Read the image and convert it to base64 for passing to the HTML template
            with open(image_path, "rb") as product_image_file:
                product_image_data = base64.b64encode(product_image_file.read()).decode("utf-8")
                print("Image data is available")

            return render_template('shadematched.html', product_name=product_name, product_price=product_price_str, product_id=product_id, product_image_data=product_image_data, product_link = product_link)
        else:
            print("Image file not found")
            return render_template('shadematched.html', product_name=product_name, product_price=product_price_str, product_id=product_id, product_image_data=None, product_link = product_link)  # Or handle the case where the image file is not found

    else:
        print("Product information not found")
        return render_template('nomatch.html')  # Or handle the case where product information is not found

@app.route('/nomatch')
def nomatch():
    return render_template('nomatch.html')

@app.route('/lipvalidation', methods=['GET', 'POST'])
def lip_validation():
    if request.files:
        if 'lipshade_file' not in request.files:
            return redirect(request.url)

        file = request.files['lipshade_file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            userhandler.set_uploaded_lipshadefinder_filename(file.filename)
            file.save(userhandler.get_uploaded_lipshadefinder_filename_path())
            mask_directory = userhandler.shaderecommender.save_predicted_mask()
            return render_template('lipvalidation.html', image_path=mask_directory)
    else:
        return render_template('upload_lipshadefinder.html')

if __name__ == '__main__':
    app.run(debug=True)