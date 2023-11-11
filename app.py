import os
import time
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_executor import Executor
from UserHandler import UserHandler
from Feedback import FeedbackModel

app = Flask(__name__, static_url_path='/static')
executor = Executor(app)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

#initialise userhandler class as a centralised platform for front, back and user experience
userhandler = UserHandler()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def redirect_home():
    return redirect('/home')

@app.route('/tutorial_virtualtryon', methods=['GET'])
def tutorial_virtualtryon():
    return render_template('tutorial_virtualtryon.html')

@app.route('/tutorial_lipshadefinder', methods=['GET'])
def tutorial_lipshadefinder():
    return render_template('tutorial_lipshadefinder.html')

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
                    
    facecounter = request.args.get('facecounter')
    if facecounter:
        face_count_dict = {}
        face_count_dict['facecount'] = userhandler.facecounter.num_faces_detected
        face_count_dict['source'] = userhandler.return_facecounter_filename()
        return jsonify(face_count_dict)
    
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
    selected_product = request.args.get('selected_product')
    if selected_product:
        destination_filename = os.path.join("static", "playground", "virtualtryon", "modified", f"{selected_product}.png")
        hexcode = product_line_dict[selected_product]['hexcode']
        image = userhandler.lipcolorizer.colorize_lips(f'#{hexcode}')
        userhandler.lipcolorizer.saveImage(image, destination_filename)
        return jsonify({selected_product: product_line_dict[selected_product]})
    else:
        return jsonify(product_line_dict)

@app.route('/home')
def home():
    return render_template('home.html')

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
            if userhandler.check_for_one_face(userhandler.get_uploaded_virtualtryon_filename_path('upload.png')):
                userhandler.set_uploaded_virtualtryon_filename()
                return render_template('virtualtryon.html')
            else:
                return render_template('failedvalidation_virtualtryon.html', image_path='upload.png')
    else:
        return render_template('upload_virtualtryon.html')

@app.route('/upload_virtualtryon', methods=['GET', 'POST'])
def upload_virtualtryon():
    return render_template('upload_virtualtryon.html')

@app.route('/upload_lipshadefinder', methods=['GET', 'POST'])
def upload_lipshadefinder():
    return render_template('upload_lipshadefinder.html')

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
            if userhandler.check_for_one_face(userhandler.get_uploaded_lipshadefinder_filename_path(file.filename)):
                userhandler.set_uploaded_lipshadefinder_filename()
                mask_directory = userhandler.shaderecommender.save_predicted_mask()
                return render_template('lipvalidation.html', image_path=mask_directory)
            else:
                return render_template('failedvalidation_lipshadefinder.html', image_path=file.filename)
            
    else:
        return render_template('upload_lipshadefinder.html')

@app.route('/training', methods=['GET', 'POST'])
def training():
    FeedbackModel('static/wrong-images', 'static/wrong-images-json').train()
    return redirect('/home')

if __name__ == '__main__':
    app.run(debug=True)