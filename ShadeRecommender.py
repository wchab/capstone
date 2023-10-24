import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import segmentation_models_pytorch as smp

class ShadeRecommender():
    def __init__(self, img_path):
        self.img_path = img_path
        self.segmodel = torch.load('./best_model.pth')
        self.output = self.predict_mask()
        self.colour_rgb  = self.extract_avg_colour()
        self.product_df = pd.read_excel('./static/lips_loreal_updated.xlsx')
        self.product_dic = self.recommend_product()
        
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

        # Create a DataFrame with the single image path
        data = pd.DataFrame({'image_path': [self.img_path]})

        # Create an instance of the ImageDataset
        dataset = ImageDataset(data,preprocessing=preprocess_input)

        # Create a DataLoader with a batch size of 1 (since you have only one image)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # access the single image as follows
        for batch in data_loader:
            image = batch
        output = self.segmodel(image)

        # Assuming output is the tensor
        output = output[0, 0].detach().cpu().numpy()  # Extract the single channel and convert to a NumPy array
        
        # Visualize the predicted mask
        plt.imshow(output, cmap='jet', vmin=0, vmax=1)  # Use an appropriate colormap, vmin, and vmax
#         visualization_image_path = 'path_to_save_visualization.png'  # Specify the path and filename
#         plt.savefig(visualization_image_path)


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



        # convert to RGB 
        avg_rgb = average_color[::-1]
        
        return avg_rgb
    
    def recommend_product(self):
        target_color = self.colour_rgb
        hex_lst = self.product_df['hex_colour'].to_list()
        
        # Function to convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])

        # Function to calculate Euclidean distance between two RGB colors
        def color_distance(color1, color2):
            return np.linalg.norm(color1 - color2)
        
        dist_lst = []
        for i in range(len(hex_lst)):
            rgb = hex_to_rgb(hex_lst[i])
            dist = color_distance(target_color, rgb)
            dist_lst.append(dist)
#         return dist_lst
        
        product_idx = dist_lst.index(min(dist_lst))
        product_row = self.product_df.loc[product_idx]
        product_dic = product_row.to_dict()
        return product_dic
    
#     def save_predicted_mask(self):
#         '''
#         generates the predicted mask for front-end
#         '''
        
#         # Load the original image
#         original_image = cv2.imread(self.img_path)  # Replace with the path to your original image
        
#         # Define a threshold value
#         threshold = 0.3

#         # Create a binary mask for values greater than the threshold
#         binary_mask = (self.output>threshold).astype(np.uint8)

#         # Resize the binary mask to match the dimensions of the original image
#         binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

#         # Create a copy of the original image
#         overlay_image = original_image.copy()

#         highlight_color = np.array([0, 240, 0])  # Green highlight in BGR format

#         alpha = 0.4
#         overlay_image[binary_mask_resized == 1] = (
#             overlay_image[binary_mask_resized == 1] * (1 - alpha)
#             + highlight_color * alpha
#         )
        
#         cv2.imwrite('path_to_overlay_image2.jpeg', overlay_image)  # Replace with the desired output path
    def save_predicted_mask(self):
        '''
        generates the predicted mask for front-end
        '''
        
        # Load the original image
        original_image = cv2.imread(self.img_path)  # Replace with the path to your original image
        
        # Define a threshold value
        threshold = 0.3

        # Create a binary mask for values greater than the threshold
        binary_mask = (self.output>threshold).astype(np.uint8)

        # Resize the binary mask to match the dimensions of the original image
        binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

        # Create a copy of the original image
        overlay_image = original_image.copy()

        highlight_color = np.array([0, 240, 0])  # Green highlight in BGR format

        alpha = 0.4
        overlay_image[binary_mask_resized == 1] = (
            overlay_image[binary_mask_resized == 1] * (1 - alpha)
            + highlight_color * alpha
        )
                
        #get filename
        filename = os.path.basename(self.img_path)

        #generate new file path
        # Desired directory path
        directory_path = "./static/user_generated_masks"

        # Create the new filepath
        mask_filepath = os.path.join(directory_path, filename)
        
        cv2.imwrite(mask_filepath, overlay_image)
        return mask_filepath
    