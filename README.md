# lip-shade-finder

## Project Description
BT4103 Business Analytics Capstone Project on Lipstick Shade Recommendation and Lipstick Shade Try-On, leveraging computer
vision and machine-learning.

## Dataset Used
-  Lip Segmentation 28K Samples (https://www.kaggle.com/datasets/olekslu/makeup-lips-segmentation-28k-samples/data)

## General Flow of Project
**Lip Shade Finder**
1. **Uploading of Celebrity Image**
- User will be prompted to upload an image of their liking will be processed by our backend lip segmentation model to detect the location of the lips in the uploaded image.
2. **Validation of Image and Face Detection**
- User will then be prompted to validate the location of the lips detected. If it is correct, our machine-learning model will do a colour extraction using K-Means clustering. If it is wrong or multiple faces are being detected, the user will be prompted to upload a new image.
3. **Lipshade Recommendation**
- The lipshade colour of the uploaded image will be extracted and ran against a K-Means clustering model to recommend a list of L'Oreal products with similar colour shades.
- `Output: List of L'Oreal lipstick Product IDs`
4. **Model Retraining**
- The uploaded images that failed the validation phase will be sent to a feedback module for further model retraining.

**Virtual Try-On Finder**
1. **Uploading of Own Image**
- This feature can be used as a function itself or a continuation function after the lipshade finder function. User can upload an image of themselves to try on the lipstick colours in L'Oreal's inventory.
2. **Machine Learning Model**
- Upon clicking on any L'Oreal lipstick on the side panel of the website, the user uploaded image will be processed by our trained model in LipColorizer.
- `Output: Applied lipstick shade on uploaded image based on lipstick shade selected`

## Folder Structure
capstone/
│   ├── FaceCounter.py
│   ├── LipColorizer.py
│   ├── ShadeRecommender.py
│   ├── UserHandler.py
│   ├── app.py
│   templates/
│   ├── home.html
│   ├── lipvalidation.html
│   ├── shadematched.html
│   ├── tutorial_lipshadefinder.html
│   ├── tutorial_virtualtryon.html
│   ├── upload_lipshadefinder.html
│   ├── upload_virtualtryon.html
│   ├── virtualtryon.html
│   ├── archive
│   │   ├── virtualtryon old.html
│   static/
│   ├── assets
│   │   ├── search.png
│   ├── images
│   │   ├── colours
│   │   │   ├── AXXXXX.png
│   │   ├── color_sensational_bricks
│   │   │   ├── AXXXXX.png
│   │   ├── color_sensational_ultimattes
│   │   │   ├── AXXXXX.png
│   │   ├── intense_volume_matte
│   │   │   ├── AXXXXX.png
│   │   ├── reds_of_worth
│   │   │   ├── AXXXXX.png
│   │   ├── sens_liq_cush
│   │   │   ├── AXXXXX.png
│   │   ├── superstay_ink_crayon
│   │   │   ├── AXXXXX.png
│   │   ├── superstay_matte_ink
│   │   │   ├── AXXXXX.png
│   │   ├── superstay_vinyl_ink
│   │   │   ├── AXXXXX.png
│   ├── playground
│   │   ├── lipshadefinder
│   │   ├── virtualtryon
│   │   │   ├── modified
│   ├── tutorial
│   │   ├── lipshadetutorial.jpg
│   │   ├── virtualtryontutorial.jpg
│   ├── user_generated_masks
│   ├── best_model.pth
│   ├── lipshades.xlsx
│   ├── shade_predictor_68_face_landmarks.dat


## Directory
[capstone/] - Contains scripts for lipshade recommendation, lip detection, face detection and flask deployment and API creation

[templates/] - Contains HTML templates for frontend development, leveraging on HTML, Javascript and CSS for functionalities and UI

[static/] - Contains static images used in frontend deployment, dataset of lipstick data

## API Documentation
`GET /api/products?=selected_product` - Based on selected_product, retrieve product data from lipstick database to get machine learning models to process images based on lipstick data.
```
Run app.py to access
localhost:5000/api/products
```
**Request Body**
Parameters  | Type | Description | Example
------------- | ------------- | ------------- | ------------- |
selected_product | string | Product ID | 'A00001'
```
{
  "A00001": {
    "color": "129 Lead", 
    "hexcode": "9F5656", 
    "product_line": "intense_volume_matte", 
    "wordsearch": "intense volume matte lipstick129 lead9f5656"
  }, 
  "A00002": {
    "color": "275 La Terra Attitude", 
    "hexcode": "C25A4E", 
    "product_line": "intense_volume_matte", 
    "wordsearch": "intense volume matte lipstick275 la terra attitudec25a4e"
  }, 
  "A00003": {
    "color": "Le Nude Admirable", 
    "hexcode": "823959", 
    "product_line": "intense_volume_matte", 
    "wordsearch": "intense volume matte lipstickle nude admirable823959"
  }
  ...
}
```

**Response Body**
{
  "A00012": {
    "color": "Le Carmin Courage", 
    "hexcode": "B70276", 
    "product_line": "intense_volume_matte", 
    "wordsearch": "intense volume matte lipstickle carmin courageb70276"
  }
}

