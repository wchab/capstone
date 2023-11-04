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

class FeedbackModel():
    def __init__(self, img_paths, mask_paths):
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.sheet = self.generate_sheet()
        self.augmented_paths = self.augment_images()
        self.segmodel = torch.load('./static/best_model.pth')
        self.output = self.train_mask()
    
    def unpack_json(self, json_paths, image_paths):
        mask_paths = []
        for n in range(len(json_paths)):
            with open(json_paths[n], 'r') as json_file:
                data = json.load(json_file)
            annotations = data["annotations"]
            image_filename = str(image_paths[n])
            image = cv2.imread(image_filename)
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

            # Save the binary mask as a PNG
            filename = os.path.basename(image_filename)
            filename = filename.split('.')[0] + "_mask.png"
            mask_directory_path = "static\generated-mask" #change the directory where validation masks are saved
            mask_filepath = os.path.join(mask_directory_path, filename)
            mask_paths.append(mask_filepath)
            cv2.imwrite(mask_filepath, binary_mask)
        return mask_paths
    
    def generate_sheet(self):
        '''
        generate Excel sheet for images and masks filenames
        '''
        list_of_an_images = sorted([i for i in Path(self.image_paths).iterdir()])
        list_of_json_paths = sorted([i for i in Path(self.mask_paths).iterdir()])
        list_of_a_masks = self.unpack_json(list_of_json_paths, list_of_an_images)
        images_masks_df = pd.DataFrame({ 'filename': list_of_an_images, 'mask': list_of_a_masks })
        images_masks_df.to_excel("augment.xlsx", index=False)  
        return pd.read_excel('augment.xlsx')
    
    def augment_images(self):
        '''
        augment images and masks for the feedback model input, outputs lists of augmented images paths
        '''
        images_list = self.sheet['filename'].to_list()
        masks_list = self.sheet['mask'].to_list()
        combined_list = [images_list, masks_list]
        transformed_image_paths = []
        transformed_mask_paths = []
        for list in combined_list:
            for image_path in list:
                pil_image = Image.open(image_path)
                
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
                    
                # Apply the transformation pipeline to the PIL image
                transformed_images = []
                for i in range(len(transformation_pipeline)):
                    transformed_images.append(transformation_pipeline[i](pil_image))
                    
                for i in range(len(multiple_transformation_pipeline)):
                    for j in range(4):
                        transformed_images.append(multiple_transformation_pipeline[i](pil_image))
                
                # Convert the tensor back to PIL format and save the images
                for i in range(len(transformed_images)):
                    transformed_image = F.to_pil_image(transformed_images[i])
                    filename = os.path.basename(image_path).split('.')[0]
                    filename = filename + "_" + str(i+1) + ".png"
                    if list == images_list:
                        image_filepath = os.path.join('static', 'augmented-images', filename) #change the directory where masks are saved
                        
                        transformed_image.save(image_filepath)
                        transformed_image_paths.append(image_filepath)
                    else:
                        mask_filepath = os.path.join('static', 'augmented-masks', filename) #change the directory where masks are saved
                        
                        transformed_image.save(mask_filepath)
                        transformed_mask_paths.append(mask_filepath)
        return [transformed_image_paths, transformed_mask_paths]
    
    def train_mask(self):
        '''
        trains images and annotated masks for feedback 
        '''
        IMG_SIZE = 256

        def resize_an_image(image_filename, new_size):
            image = cv2.imread(image_filename)
            resized_image = cv2.cvtColor(cv2.resize(image, (new_size, new_size)), cv2.COLOR_BGR2RGB)
            return resized_image
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        BACKBONE = 'resnet34'
        self.segmodel.to(device)
        preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name=BACKBONE, pretrained='imagenet')

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

        # Create a DataFrame containing the training data for feedback
        augmented_images_path = self.augmented_paths
        data = pd.DataFrame({'image_paths': augmented_images_path[0], 'mask_paths': augmented_images_path[1]})
        
        segmodel = self.segmodel 

        #Split dataset into training and test sets
        X_train, X_valid = train_test_split(data, test_size=0.3,random_state=42)

        X_train.reset_index(drop=True, inplace=True)
        X_valid.reset_index(drop=True, inplace=True)

        train_data = ImageDataset(X_train, preprocessing=preprocess_input)
        valid_data = ImageDataset(X_valid, preprocessing=preprocess_input)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False)

        criterion = utils.losses.BCELoss()
        metrics = [utils.metrics.IoU(), ]
        optimizer = torch.optim.Adam(segmodel.parameters(), lr=0.001)
        
        train_epoch = utils.train.TrainEpoch(
            segmodel,
            loss=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True
        )

        valid_epoch = utils.train.ValidEpoch(
            segmodel,
            loss=criterion,
            metrics=metrics,
            device=device,
            verbose=True
        )

        def objective(trial):
            max_score = 0
            # Define hyperparameter search space
            lr = trial.suggest_float("lr", 1e-5, 1e-2)
            batch_size = trial.suggest_int("batch_size", 4, 64)
            num_epochs = 2  # Increase the number of epochs

            # Create and train the model with the suggested hyperparameters
            optimizer = torch.optim.Adam(segmodel.parameters(), lr=lr)

            for epoch in range(num_epochs):
                print("In Function")
                print("Epoch", epoch)
                train_logs = train_epoch.run(train_loader)

            # After training, evaluate the model on the validation set
            valid_logs = valid_epoch.run(valid_loader)

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(segmodel, './best_model.pth')

            return valid_logs["iou_score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2)  # You can adjust the number of trials

        # Get the best hyperparameters from the study
        best_params = study.best_params

        # Use the best hyperparameters to train the final model
        best_lr = best_params["lr"]
        best_batch_size = best_params["batch_size"]
        num_epochs = 2  # Increase the number of epochs further

        # Create and train the final model with the best hyperparameters
        # segmodel = YourModel()  # Define your model
        optimizer = torch.optim.Adam(segmodel.parameters(), lr=best_lr)

        for epoch in range(num_epochs):
            print("Out of function")
            print("Epoch", epoch)
            train_logs = train_epoch.run(train_loader)

        # Evaluate the final model on the test set
        test_logs = valid_epoch.run(valid_loader)
        print(f"Test IoU Score: {test_logs['iou_score']}")

FeedbackModel('static\wrong-images', 'static\mask-json')