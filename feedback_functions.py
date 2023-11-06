import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from PIL import Image
import random
import pandas as pd
import os
import cv2
from pathlib import Path
import json
import numpy as np
import openpyxl
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from segmentation_models_pytorch import utils
import optuna
import matplotlib.pyplot as plt

def unpack_json_mask(json_path, img_path):
    json_file_path = json_path

    # Open and read the JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    annotations = data["annotations"]
    image = cv2.imread(img_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw masks for the annotations
    for annotation in annotations:
        segmentation = annotation["segmentation"][0]  # Take the first segment
        points = np.array(segmentation, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], (255, 255, 255))  # Fills the mask with white for the annotation

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Convert the result to a binary mask
    binary_mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    
    mask_filename = os.path.basename(img_path).split('.')[0]
    mask_filename = mask_filename + "_mask" + ".png"
    # Save the binary mask as a PNG
    cv2.imwrite(mask_filename, binary_mask)
    return mask_filename

def augment_image_and_mask(image_path, mask_path):
    '''
    Augment an image and its corresponding mask and return lists of augmented image and mask paths
    '''
    pil_image = Image.open(image_path)
    pil_mask = Image.open(mask_path)

    # Define the transformation pipeline for augmentation
    transformation_list = [
        v2.RandomHorizontalFlip(p=1), 
        v2.RandomVerticalFlip(p=1), 
        v2.GaussianBlur(kernel_size=9),
        v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.75, 1.333), interpolation=2),
        v2.Pad(padding=10),
        v2.Pad(padding=20),
        v2.Pad(padding=30),
        v2.Pad(padding=40),
        v2.Pad(padding=50),
        v2.RandomGrayscale(1),
        v2.RandomPerspective(distortion_scale=0.6, p=1.0),
        v2.RandomAutocontrast(0.75),
        v2.RandomEqualize(),
    ]

    multiple_transformation_list = [
        v2.RandomRotation(degrees=random.uniform(15, 60)),
        v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        v2.ColorJitter(brightness=random.uniform(0.25, 1), contrast=random.uniform(0.25, 1), saturation=random.uniform(0.25, 1), hue=random.uniform(0.25, 0.5)),
    ]

    transformation_pipeline = []
    multiple_transformation_pipeline = []

    for i in range(len(transformation_list)):
        transformation_pipeline.append(v2.Compose([
            transformation_list[i],
            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        ]))

    for i in range(len(multiple_transformation_list)):
        multiple_transformation_pipeline.append(v2.Compose([
            multiple_transformation_list[i],
            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        ]))

    # Apply the transformation pipeline to the PIL image and mask
    transformed_images = []
    transformed_masks = []

    for i in range(len(transformation_pipeline)):
        transformed_images.append(transformation_pipeline[i](pil_image))
        transformed_masks.append(transformation_pipeline[i](pil_mask))

    for i in range(len(multiple_transformation_pipeline)):
        for j in range(4):
            transformed_images.append(multiple_transformation_pipeline[i](pil_image))
            transformed_masks.append(multiple_transformation_pipeline[i](pil_mask))

    # Convert the tensor back to PIL format and save the images
    transformed_image_paths = []
    transformed_mask_paths = []

    for i in range(len(transformed_images)):
        transformed_image = F.to_pil_image(transformed_images[i])
        transformed_mask = F.to_pil_image(transformed_masks[i])

        img_filename = os.path.basename(image_path).split('.')[0]
        img_filename = img_filename + "_" + str(i + 1) + ".jpeg"
        
        mask_filename = os.path.basename(mask_path).split('.')[0]
        mask_filename = mask_filename + "_" + str(i + 1) + ".png"

        image_filepath = os.path.join('./static', 'augmented_images', img_filename)
        mask_filepath = os.path.join('./static', 'augmented_masks', mask_filename)

        transformed_image.save(image_filepath)
        transformed_mask.save(mask_filepath)

        transformed_image_paths.append(image_filepath)
        transformed_mask_paths.append(mask_filepath)
    df = pd.DataFrame({'filename': transformed_image_paths, 'mask': transformed_mask_paths})
    print(transformed_image_paths)
    print(transformed_mask_paths)

        # Save the DataFrame to an Excel file
    df.to_excel('feedback_image_mask_paths.xlsx', index=False)

#     return [transformed_image_paths, transformed_mask_paths]


# unpack_json_mask('panther.json', 'panther.jpeg')
