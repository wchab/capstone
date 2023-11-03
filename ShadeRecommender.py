import CIEDE2000 as calc_delta
import numpy as np
import os
import numpy as np
import pandas as pd
import cv2
import torch
import segmentation_models_pytorch as smp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ShadeRecommender():
    def __init__(self, img_path):
        self.img_path = img_path
        self.segmodel = torch.load('best_model.pth')
        self.output = self.predict_mask()
        self.binary_mask = self.store_binary_mask()
        self.product_df = pd.read_excel('lipshades.xlsx')

        
    def predict_mask(self):
        '''
        takes in img_path and generates a predicted mask 
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        BACKBONE = 'resnet34'
        self.segmodel.to(device)
        preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name=BACKBONE, pretrained='imagenet')

        #helpers
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

        return output

    def store_binary_mask(self):
        '''
        function to store binary mask used by other methods
        '''
        # Define a threshold value
        threshold = 0.5
        
        # Create a binary mask for values greater than the threshold
        binary_mask = (self.output>threshold).astype(np.uint8)

        return binary_mask
    
    def recommend_products(self):
        '''

        '''
        try:
            original_image = cv2.imread(self.img_path) 

            # Resize the binary mask to match the dimensions of the original image
            binary_mask_resized = cv2.resize(self.binary_mask, (original_image.shape[1], original_image.shape[0]))

            mask = (binary_mask_resized == 1)

            # Use the mask to extract the pixels from the segmented region
            segmented_pixels = original_image[mask]


            colour_lst = []

            # Calculate the average color for the segmented pixels]
            average_color = np.mean(segmented_pixels, axis=0)
            # convert to RGB 
            avg_rgb = average_color[::-1]
            colour_lst.append(avg_rgb)

            #dominant colours
            segmented_pixels_rgb = list(map(lambda row: [row[2], row[1], row[0]], segmented_pixels))


            silhouette_scores = []
            for n_clusters in range(2, 11):
                kmeans = KMeans(n_clusters=n_clusters, n_init = 10, random_state=0)
                kmeans_labels = kmeans.fit_predict(segmented_pixels_rgb)
                silhouette_avg = silhouette_score(segmented_pixels_rgb, kmeans_labels)
                silhouette_scores.append(silhouette_avg)

            optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

            kmeans = KMeans(n_clusters=optimal_num_clusters, random_state = 0)
            kmeans.fit_predict(segmented_pixels_rgb)
    #================================================VISUALISATIONS=======================================       
            #             #score graph
    #             # Create an array of cluster sizes (2 to 10 in your case)
    #             cluster_sizes = range(2, 11)

    #             # Plot the silhouette scores for different cluster sizes
    #             plt.figure(figsize=(8, 6))
    #             plt.plot(cluster_sizes, silhouette_scores, marker='o', linestyle='-')
    #             plt.title('Silhouette Score vs. Number of Clusters')
    #             plt.xlabel('Number of Clusters')
    #             plt.ylabel('Average Silhouette Score')
    #             plt.grid(True)
    #             plt.show()
    #             #score graph
    #             #3d graph
    #             fig = plt.figure(figsize = (15,15))            
    #             ax = fig.add_subplot(111, projection='3d')


    #             # Define unique colors for each cluster
    #             colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    #             # Plot each point with a color corresponding to its cluster assignment
    #             scatter_plots = []
    #             for i in range(len(segmented_pixels_rgb)):
    #                 scatter = ax.scatter(segmented_pixels_rgb[i][0], segmented_pixels_rgb[i][1], segmented_pixels_rgb[i][2], c=colors[kmeans_labels[i]], marker='o')
    #                 scatter_plots.append(scatter)

    #             # Customize the plot
    #             ax.set_xlabel('Red')
    #             ax.set_ylabel('Green')
    #             ax.set_zlabel('Blue')

    #             # Create a legend
    #             legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', markerfacecolor=colors[i]) for i in range(optimal_num_clusters)]
    #             ax.legend(handles=legend_elements, title='Clusters')

    #             plt.show()

    #             #3d graph
    #================================================VISUALISATIONS=======================================  
            colour_lst.extend(kmeans.cluster_centers_)
            target_colors = colour_lst
            hex_lst = self.product_df['hexcode'].to_list()

            # Function to convert hex to RGB
            def hex_to_rgb(hex_color):
                return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])

            # # Function to calculate Euclidean distance between two RGB colors
            # def colour_distance(color1, color2):

            #     return np.linalg.norm(color1 - color2)
            
            def rgb_to_lab(rgb_lst):
                # Create a single-pixel image with the RGB color
                rgb_array = np.array([rgb_lst], dtype=np.uint8)
                image = rgb_array.reshape(1, 1, 3)

                # Convert the single-pixel image to the Lab color space
                lab_color = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

                # Extract the L, a, and b components
                L, a, b = cv2.split(lab_color)
                return (L[0,0], a[0][0], b[0,0])
            
            recommeded_product_ids = []
            #=============CIELAB method======================
            for colour in target_colors:
                target_lab = rgb_to_lab(colour)
                delta_lst = []
                for i in range(len(hex_lst)):
                    rgb = hex_to_rgb(str(hex_lst[i]))
                    lab = rgb_to_lab(rgb)
                    delta = calc_delta.CIEDE2000(target_lab, lab)
                    delta_lst.append(delta)

                product_idx = delta_lst.index(min(delta_lst))
                product_id = self.product_df.loc[product_idx, "product_id"]
                if product_id not in recommeded_product_ids:
                    recommeded_product_ids.append(product_id)                
            #=============END CIELAB method======================

            #=============euclidean distance method======================
    #         for colour in target_colors:
    #             dist_lst = []
    #             for i in range(len(hex_lst)):
    #                 rgb = hex_to_rgb(str(hex_lst[i]))
    #                 dist = colour_distance(colour, rgb)
    #                 dist_lst.append(dist)

    #             product_idx = dist_lst.index(min(dist_lst))
    #             product_id = self.product_df.loc[product_idx, "product_id"]
    #             if product_id not in recommeded_product_ids:
    #                 recommeded_product_ids.append(product_id)
            #=============END euclidean distance method======================
            return recommeded_product_ids          

        except Exception as e:
            print(" Lips were not detected! Please upload another image! I will learn from this mistake!")
    
    def save_predicted_mask(self):
        '''
        front-end to run this specifically to generate and save generated mask
        '''
        
        # Load the original image
        original_image = cv2.imread(self.img_path)  # Replace with the path to your original image

        # Resize the binary mask to match the dimensions of the original image
        binary_mask_resized = cv2.resize(self.binary_mask, (original_image.shape[1], original_image.shape[0]))

        # Create a copy of the original image
        overlay_image = original_image.copy()

        highlight_color = np.array([0, 240, 0])  # Green highlight in BGR format

        alpha = 0.4
        overlay_image[binary_mask_resized == 1] = (
            overlay_image[binary_mask_resized == 1] * (1 - alpha)
            + highlight_color * alpha
        )
                
        filename = os.path.basename(self.img_path)

        # Desired directory path
        directory_path = "./static/user_generated_masks"

        # Create the new filepath
        mask_filepath = os.path.join(directory_path, filename)
        
        cv2.imwrite(mask_filepath, overlay_image)
        return mask_filepath