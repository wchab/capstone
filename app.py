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
    recommended_products = request.args.get('recommended_products')
    if recommended_products:
        recommended_line_dict = {}
        try: 
            recommended_products = userhandler.shaderecommender.recommend_products()
            for product in recommended_products:
                recommended_line_dict[product] = product_line_dict[product]
            return jsonify(recommended_line_dict)
        except:
            return {}
            
    else:
        return jsonify(product_line_dict)

@app.route('/home')
def home():
    return render_template('home.html')
    
def process_image(filename):
    df = pd.read_excel('./static/lipshades.xlsx')
    df['hexcode'] = df['hexcode'].map(lambda x: str(x))
    hexcode_dict = dict(zip(df['product_id'], df['hexcode']))
    for key in hexcode_dict.keys():
        destination_filename = os.path.join("static", "playground", "virtualtryon", "modified", f"{key}.png")
        image = userhandler.lipcolorizer.colorize_lips(f'#{hexcode_dict[key]}')
        userhandler.lipcolorizer.saveImage(image, destination_filename)
        print(f"{key} processed")

@app.route('/virtualtryon', methods=['GET', 'POST'])
def virtualtryon():
    if request.files:
        if 'virtualtryon_file' not in request.files:
            return redirect(request.url)

        file = request.files['virtualtryon_file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file.save(userhandler.get_uploaded_virtualtryon_filename_path('upload.png'))
            userhandler.set_uploaded_virtualtryon_filename()
            executor.submit(process_image(file.filename))
            return render_template('virtualtryon.html')
    else:
        return render_template('virtualtryon.html')

@app.route('/upload_virtualtryon', methods=['GET', 'POST'])
def upload_virtualtryon():
    return render_template('upload_virtualtryon.html')

@app.route('/upload_lipshadefinder', methods=['GET', 'POST'])
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

@app.route('/shadematched')
def match_product():
    try:
        recommended_products = userhandler.shaderecommender.recommend_products()
        return render_template('shadematched.html', recommended_products=recommended_products)
    except:
        return render_template('shadematched.html')
 
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
            file.save(userhandler.get_uploaded_lipshadefinder_filename_path(file.filename))
            userhandler.set_uploaded_lipshadefinder_filename()
            mask_directory = userhandler.shaderecommender.save_predicted_mask()
            return render_template('lipvalidation.html', image_path=mask_directory)
    else:
        return render_template('upload_lipshadefinder.html')

if __name__ == '__main__':
    app.run(debug=True)