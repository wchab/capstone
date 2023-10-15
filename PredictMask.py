import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import cv2
import torch
from torch import nn, optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

class PredictImage():
    def __init__(self):
        self.image_filename = None
        print("Locating Lips...")
        pass

    def predict_mask(self, img_path):
        if '/' in img_path:
            self.image_filename = img_path.split("/")[-1]
        segmodel = torch.load('./best_model.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        BACKBONE = 'resnet34'
        segmodel.to(device)
        preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name=BACKBONE, pretrained='imagenet')
        
        #helpers
        # Define the image size you want
        IMG_SIZE = 256

        def resize_an_image(image_filename, new_size):
            image = cv2.imread(image_filename)
            resized_image = cv2.cvtColor(cv2.resize(image, (new_size, new_size)), cv2.COLOR_BGR2RGB)
            return resized_image

        class ImageDataset(torch.utils.data.Dataset):

            def __init__(self, data, preprocessing=None):
                self.data = data
                self.preprocessing = preprocessing
                self.image_paths = self.data.iloc[:, 0]
                self.data_len = len(self.data.index)

            def __len__(self):
                return self.data_len

            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                img = resize_an_image(image_path, IMG_SIZE)
                img = img.astype(float)

                if self.preprocessing:
                    img = self.preprocessing(img)
                    img = torch.as_tensor(img)
                else:
                    img = torch.as_tensor(img)
                    img /= 255.0

                img = img.permute(2, 0, 1)

                return img.float()

        # Create a DataFrame with the single image path
        data = pd.DataFrame({'image_path': [img_path]})

        # Create an instance of the ImageDataset
        dataset = ImageDataset(data,preprocessing=preprocess_input)

        # Create a DataLoader with a batch size of 1 (since you have only one image)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # You can access the single image as follows
        for batch in data_loader:
            image = batch

        output = segmodel(image)
    
        output_array = output[0, 0].detach().cpu().numpy() 

        output_image_path = os.path.join("processed", f"processed_{self.image_filename}")

        # Convert the predicted mask to the range [0, 255] and change data type to uint8
        output_image = (output_array * 255).astype(np.uint8)
        # Save the image
        cv2.imwrite(output_image_path, output_image)
        print(output_image_path)
        return output_image_path
