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

#Prepare dataset class and wrap it into DataLoader
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
        segmodel = torch.load('./best_model.pth')
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



        # Load the pre-saved model if available

        segmodel = torch.load('./best_model.pth')


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
        optimizer = optim.Adam(segmodel.parameters(), lr=best_lr)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=best_batch_size, shuffle=True)

        for epoch in range(num_epochs):
            print("Out of function")
            print("Epoch", epoch)
        train_logs = train_epoch.run(train_loader)

        # Evaluate the final model on the test set
        test_logs = valid_epoch.run(valid_loader)
        print(f"Test IoU Score: {test_logs['iou_score']}")