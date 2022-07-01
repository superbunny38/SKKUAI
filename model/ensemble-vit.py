import copy
from torchvision import models
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import os
import re
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import normalize
import PIL.Image as Image
from self_attention_cv import ResNet50ViT, ViT

''' dataset 준비 '''
df=pd.read_csv('./train/grade_labels.csv')
labelencoder=preprocessing.LabelEncoder()
df['Grade']=labelencoder.fit(['1++','1+','1','2','3']).transform(df['grade'])
train_len=int(len(df)*0.9)
val_idx=int(train_len*0.8)

df_train=df.iloc[:train_len]
df_val=pd.DataFrame.copy(df.iloc[val_idx:train_len]).reset_index(drop=True)
df_test=pd.DataFrame.copy(df.iloc[train_len:]).reset_index(drop=True)

csv_datasets = {'train': df_train, 'val': df_val, 'test': df_test}

class Dataset(Dataset):

    def __init__(self, data, path, transform=None):
        self.data=data # data = csv file
        self.path=path # data directory
        self.transform=transform

    def __getitem__(self, idx):
        file_name = self.data['imname'][idx] # 인덱스에 맞는 파일명 확인
        img = Image.open(self.path+file_name)
        label = self.data['Grade'][idx]
        label = torch.tensor(label, dtype=torch.int64)

        if self.transform:
            data = self.transform(img)

        return data, label
    
    def __len__(self):
        return len(self.data)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((272,272)),
        transforms.RandomRotation(15,),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
     'test': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
}

data_dir = './train/images/'

image_datasets={x: Dataset(csv_datasets[x],data_dir,transform=data_transforms[x])
              for x in ['train', 'val', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


''' Vision Transformer 용 '''
# model_ft = ResNet50ViT(img_dim=256, pretrained_resnet=True, num_classes=5, dim_linear_block=256, dim=256)
model_ft = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=5,dim=256)


''' train 준비 '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_ft = model_ft.to(device)

''' ensemble '''
from torchensemble import VotingClassifier  # voting is a classic ensemble strategy


# Define the ensemble
ensemble = VotingClassifier(
    estimator=model_ft,               # here is your deep learning model
    n_estimators=5,                        # number of base estimators
)
# Set the criterion
criterion = nn.CrossEntropyLoss()           # training objective
ensemble.set_criterion(criterion)

# Set the optimizer
ensemble.set_optimizer(
    "SGD",                                 # type of parameter optimizer
    lr=0.01,                       # learning rate of parameter optimizer
    weight_decay=0.0001,              # weight decay of parameter optimizer
)

# Set the learning rate scheduler
ensemble.set_scheduler(
    "CosineAnnealingLR",                    # type of learning rate scheduler
    T_max=30,                           # additional arguments on the scheduler
)

ensemble.to(device=device)

# Train the ensemble
ensemble.fit(
    dataloaders['train'],
    epochs=20,                          # number of training epochs
)

''' test '''
test_df = pd.read_csv('./test/test_images.csv')
test_df['Grade']=[0 for i in range(len(test_df))]

real_test_dataset = Dataset(test_df,'./test/images/',transform=data_transforms['test'])

index2grade={2:'1++',1:'1+',0:'1',3:'2',4:'3'}

# Evaluate the ensemble
acc = ensemble.evaluate(dataloaders['test'])         # testing accuracy
print("acc:",acc)

torch.save(ensemble.state_dict(), './vit_Ensemble.pt')

for data, label in real_test_dataset:
    data = data.view(1,data.shape[0],data.shape[1],data.shape[2])
    index = torch.argmax(ensemble.predict(data))
    print(index2grade[index.item()])