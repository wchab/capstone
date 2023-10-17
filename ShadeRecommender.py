import matplotlib.pyplot as plt
import numpy as np
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch

from torch import nn, optim
from segmentation_models_pytorch import utils
from pathlib import Path
from sklearn.model_selection import train_test_split
    
class ShadeRecommender():
    def __init__(self, img_path):
        self.img_path = img_path
        self.segmodel = torch.load('./best_model.pth')
        self.output = self.predict_mask()
        self.colour_rgb  = self.extract_avg_colour()
        
    def predict_mask(self):
        '''
        takes in img_path and generates a predicted mask 
        '''
    #     segmodel = torch.load('./best_model.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        BACKBONE = 'resnet34'
        self.segmodel.to(device)
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
        # Your single image file path
    #     image_path = 'lol.jpeg'

        # Create a DataFrame with the single image path
        data = pd.DataFrame({'image_path': [self.img_path]})

        # Create an instance of the ImageDataset
        dataset = ImageDataset(data,preprocessing=preprocess_input)

        # Create a DataLoader with a batch size of 1 (since you have only one image)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # You can access the single image as follows
        for batch in data_loader:
            image = batch
        output = self.segmodel(image)

        # Assuming output is the tensor you provided
        output = output[0, 0].detach().cpu().numpy()  # Extract the single channel and convert to a NumPy array

#         # Visualize the predicted mask
        plt.imshow(output, cmap='jet', vmin=0, vmax=1)  # Use an appropriate colormap, vmin, and vmax
        plt.colorbar()

    #     # Save the predicted mask as an image
    #     output_image_path = 'path_to_save_output_mask.png'  # Specify the path and filename for the output image

    #     # Convert the predicted mask to the range [0, 255] and change data type to uint8
    #     output_image = (output * 255).astype(np.uint8)
    #     # Save the image
    #     cv2.imwrite(output_image_path, output_image)

    #     print(f"Predicted mask saved as {output_image_path}")
    #     # Load the original image
    #     original_image = cv2.imread(img_path)  # Replace with the path to your original image

        return output
    
    def extract_avg_colour(self):
        '''
        function to reformat identified lip region back onto original image and extract average pixel colour
        '''
        original_image = cv2.imread(self.img_path)  # Replace with the path to your original image
        
        # Define a threshold value
        threshold = 0.3
        
        # Create a binary mask for values greater than the threshold
        binary_mask = (self.output>threshold).astype(np.uint8)

        # Resize the binary mask to match the dimensions of the original image
        binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

        mask = (binary_mask_resized == 1)

        # Use the mask to extract the pixels from the segmented region
        segmented_pixels = original_image[mask]

        # Calculate the average color for the segmented pixels
        average_color = np.mean(segmented_pixels, axis=0)

        # The 'average_color' variable will contain the average color as an array of BGR values

        # If you want to convert it to RGB (typical for visualization), you can do:
        avg_rgb = average_color[::-1]
        
        #convert to hex
        
        return avg_rgb
    
#     def recommend_product(self):
        
        
        
    
    
