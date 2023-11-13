
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model

class Helpers:
    def dataframe_to_arrays(dataframe,normalize = True):
        # Make a copy of the original dataframe
        df = dataframe.copy(deep=True)
        input_cols = df.columns.values[1:]
        output_cols = df.columns.values[:1]
        # Convert non-numeric categorical columns to numbers
        cat_cols = df.select_dtypes(include=object).columns.values
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes #give numerical codes
        if normalize:
            for col in input_cols:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        # Extract input & outputs as numpy arrays
        X = df[input_cols].to_numpy()
        y = df[output_cols].to_numpy().reshape(-1,1)
        return X,y

    def accuracy(preds, targets):
        return torch.tensor(torch.sum(preds.round() == targets).item() / len(preds))

class Titanic_Model_1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions 
        y_pred = self(inputs)        
        # Calcuate loss
        loss = F.mse_loss(y_pred, targets)                          
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        y_pred = self(inputs)       
        # Calcuate loss
        loss = F.mse_loss(y_pred, targets)                           
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() 
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch+1,result['val_loss']))