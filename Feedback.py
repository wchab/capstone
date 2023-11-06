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
from torch import nn, optim

class FeedbackModel():
    def __init__(self, images_folder, json_folder):
        self.images_folder = images_folder
        self.json_folder = json_folder

    def preprocessing(self):
        print(self.images_folder)
        for filename in os.listdir(self.images_folder):
            if 'DS_Store' not in filename:
                filename_title = filename.split('.')[0]
                print(f"===== PREPROCESSING ON {filename} ====")
                if f'{filename_title}.json' not in os.listdir(self.json_folder):
                    print("Error: Upload manual generated COCO JSON file from makesense.ai")
                    print("Ensure that image and json files are named the same")
                else:
                    json_path = os.path.join('static', 'wrong-images-json', f'{filename_title}.json')
                    img_path = os.path.join('static', 'wrong-images', filename)
                    mask_path = self.unpack_json_mask(json_path, img_path)
                    self.augment_image_and_mask(img_path, mask_path)
                    Retrain(self.image_mask_df)
                    
    def unpack_json_mask(self, json_path, img_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        annotations = data["annotations"]
        image = cv2.imread(img_path)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for annotation in annotations:
            segmentation = annotation["segmentation"][0]
            points = np.array(segmentation, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [points], (255, 255, 255))

        result = cv2.bitwise_and(image, image, mask=mask)
        binary_mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
        
        mask_filename = os.path.basename(img_path).split('.')[0]
        mask_filename = mask_filename + ".png"

        cv2.imwrite(os.path.join('static', 'wrong-generated-mask', mask_filename), binary_mask)
        return os.path.join('static', 'wrong-generated-mask', mask_filename)

    def augment_image_and_mask(self, image_path, mask_path):
        '''
        Augment an image and its corresponding mask and return lists of augmented image and mask paths
        '''
        pil_image = Image.open(image_path)
        pil_mask = Image.open(mask_path)

        # Define the transformation pipeline for augmentation
        transformation_list = [
            v2.RandomHorizontalFlip(p=1), 
            v2.RandomVerticalFlip(p=1), 
            # v2.GaussianBlur(kernel_size=9),
            # v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.75, 1.333), interpolation=2),
            # v2.Pad(padding=10),
            # v2.Pad(padding=20),
            # v2.Pad(padding=30),
            # v2.Pad(padding=40),
            # v2.Pad(padding=50),
            # # v2.RandomGrayscale(1),
            # v2.RandomPerspective(distortion_scale=0.6, p=1.0),
            # v2.RandomAutocontrast(0.75),
            # v2.RandomEqualize()
        ]

        multiple_transformation_list = [
            v2.RandomRotation(degrees=random.uniform(15, 60)),
            v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            v2.ColorJitter(brightness=random.uniform(0.25, 1), contrast=random.uniform(0.25, 1), saturation=random.uniform(0.25, 1), hue=random.uniform(0.25, 0.5))
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

        transformed_image_paths = []
        transformed_mask_paths = []

        for i in range(len(transformed_images)):
            transformed_image = F.to_pil_image(transformed_images[i])
            transformed_mask = F.to_pil_image(transformed_masks[i])

            img_filename = os.path.basename(image_path).split('.')[0]
            img_filename = img_filename + "_" + str(i + 1) + ".png"
            
            mask_filename = os.path.basename(mask_path).split('.')[0]
            mask_filename = mask_filename + "_" + str(i + 1) + ".png"

            image_filepath = os.path.join('./static', 'augmented-images', img_filename)
            mask_filepath = os.path.join('./static', 'augmented-masks', mask_filename)

            transformed_image.save(image_filepath)
            transformed_mask.save(mask_filepath)

            transformed_image_paths.append(image_filepath)
            transformed_mask_paths.append(mask_filepath)
        self.image_mask_df = pd.DataFrame({'filename': transformed_image_paths, 'mask': transformed_mask_paths})
        print(self.image_mask_df)
        pass

class Retrain():
    def __init__(self, df_lips_images_tuning):            
        IMG_SIZE = 256

        def resize_an_image_and_mask(image_filename, mask_filename, new_size):
            image = cv2.imread(image_filename)
            mask = cv2.imread(mask_filename)
            
            resized_image = cv2.cvtColor(cv2.resize(image, (new_size, new_size)), cv2.COLOR_BGR2RGB)
            resized_mask = cv2.cvtColor(cv2.resize(mask, (new_size, new_size)), cv2.COLOR_BGR2RGB)
            
            return resized_image, resized_mask

        class LipsDataset(torch.utils.data.Dataset):
            
            def __init__(self, data, preprocessing=None):
                
                self.data = data
                self.preprocessing = preprocessing
                
                self.images_paths = self.data.iloc[:, 0]
                self.masks_paths = self.data.iloc[:, 1]
                
                self.data_len = len(self.data.index)
                
            def __len__(self):
                return self.data_len

            def __getitem__(self, idx):
                
                img_path = self.images_paths[idx]
                mask_path = self.masks_paths[idx]
                
                img, mask = resize_an_image_and_mask(img_path, mask_path, IMG_SIZE)
                
                img = img.astype(float)
                
                if self.preprocessing:
                    img = self.preprocessing(img)
                    img = torch.as_tensor(img)
                else:
                    img = torch.as_tensor(img)
                    img /= 255.0
                
                img = img.permute(2, 0, 1)
                cls_mask_1 = mask[..., 1]
                cls_mask_1 = np.where(mask > 50, 1, 0)[:,:,1]
                cls_mask_1 = cls_mask_1.astype('float')
                masks = [cls_mask_1]
                masks = torch.as_tensor(masks, dtype=torch.float)
                
                return img.float(), masks
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device
        segmodel = torch.load('static/best_model.pth')
        segmodel.to(device)

        #Split dataset into training and test sets
        X_train, X_valid = train_test_split(df_lips_images_tuning, test_size=0.3,random_state=42)

        X_train.reset_index(drop=True, inplace=True)
        X_valid.reset_index(drop=True, inplace=True)
        BACKBONE = 'resnet34'

        preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name=BACKBONE, pretrained='imagenet')
        train_data = LipsDataset(X_train, preprocessing=preprocess_input)
        valid_data = LipsDataset(X_valid, preprocessing=preprocess_input)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=False)
        criterion = utils.losses.BCELoss()
        metrics = [utils.metrics.IoU(), ]
        optimizer = optim.Adam(segmodel.parameters(), lr=0.001)

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
            num_epochs = 10  # Increase the number of epochs

            # Create and train the model with the suggested hyperparameters
            # segmodel = YourModel()  # Define your model
            optimizer = optim.Adam(segmodel.parameters(), lr=lr)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

            for epoch in range(num_epochs):
                print("In Function")
                print("Epoch", epoch)
                train_logs = train_epoch.run(train_loader)

            # After training, evaluate the model on the validation set
            valid_logs = valid_epoch.run(valid_loader)

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                print('hello')
                torch.save(segmodel, './static/best_model.pth')

            return valid_logs["iou_score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2)  # You can adjust the number of trials

        # Get the best hyperparameters from the study
        best_params = study.best_params

        # Use the best hyperparameters to train the final model
        best_lr = best_params["lr"]
        best_batch_size = best_params["batch_size"]
        num_epochs = 10  # Increase the number of epochs further

        # Create and train the final model with the best hyperparameters
        # segmodel = YourModel()  # Define your model
        optimizer = optim.Adam(segmodel.parameters(), lr=best_lr)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=best_batch_size, shuffle=True)

        for epoch in range(num_epochs):
            print("Out of function")
            print("Epoch", epoch)
        train_logs = train_epoch.run(train_loader)

        # Evaluate the final model on the test set
        test_logs = valid_epoch.run(valid_loader)
        print(f"Test IoU Score: {test_logs['iou_score']}")
            
FeedbackModel('static/wrong-images', 'static/wrong-images-json').preprocessing()