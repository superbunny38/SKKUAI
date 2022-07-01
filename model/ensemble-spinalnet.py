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
        transforms.Resize((256,256)),
        transforms.RandomRotation(15,),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[64.28836667, 32.85761432, 31.56599267], std=[56.11941082, 38.43741002, 38.49034872])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[64.28836667, 32.85761432, 31.56599267], std=[56.11941082, 38.43741002, 38.49034872])
    ]),
     'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[64.28836667, 32.85761432, 31.56599267], std=[56.11941082, 38.43741002, 38.49034872])
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




model_ft = models.wide_resnet50_2(pretrained=True)
num_ftrs = model_ft.fc.in_features

half_in_size = round(num_ftrs/2)
layer_width = 200 #Small for Resnet, large for VGG
Num_class=5

class SpinalNet_ResNet(nn.Module):
    def __init__(self):
        super(SpinalNet_ResNet, self).__init__()
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Linear(half_in_size, layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Linear(layer_width*4, Num_class),)
        
    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,half_in_size:2*half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,half_in_size:2*half_in_size], x3], dim=1))
        
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        
        x = self.fc_out(x)
        return x
 

'''
Changing the fully connected layer to SpinalNet or VGG or ResNet
'''

#model_ft.fc = nn.Linear(num_ftrs, 100)
model_ft.fc = SpinalNet_ResNet() #SpinalNet_VGG

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    test_token=0

    earlystop=EarlyStopping(path='./SpinalNet_earlystop.pt')
    is_earlystop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            '''
            Test when a better validation result is found
            '''
            if test_token ==0 and phase == 'test':
                continue
            test_token =0
            
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                test_token =1

            if phase == 'val' and earlystop(epoch_acc, model):
                print("Early stopped!")
                is_earlystop = True
                break

        print()
        if is_earlystop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

''' train 준비 '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ft = model_ft.to(device)
# criterion = nn.CrossEntropyLoss()
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001,momentum=0.9)
# # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# ''' train '''
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
# torch.save(model_ft.state_dict(),"./SpinalNet.pt")

# ''' test '''
test_df = pd.read_csv('./test/test_images.csv')
test_df['Grade']=[0 for i in range(len(test_df))]

real_test_dataset = Dataset(test_df,'./test/images/',transform=data_transforms['test'])

index2grade={2:'1++',1:'1+',0:'1',3:'2',4:'3'}

# model_ft.eval()
# for data, label in real_test_dataset:
#     data = data.view(1,data.shape[0],data.shape[1],data.shape[2])
#     data = data.to(device)
#     output = model_ft(data)
#     _, preds = torch.max(output, 1)
#     print(index2grade[preds.item()])

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
    T_max=20,                           # additional arguments on the scheduler
)

# Train the ensemble
ensemble.fit(
    dataloaders['train'],
    epochs=20,                          # number of training epochs
)

# Evaluate the ensemble
acc = ensemble.evaluate(dataloaders['test'])         # testing accuracy
print("acc:",acc)

torch.save(ensemble.state_dict(), './last_spinalnet_Ensemble.pt')

for data, label in real_test_dataset:
    data = data.view(1,data.shape[0],data.shape[1],data.shape[2])
    index = torch.argmax(ensemble.predict(data))
    print(index2grade[index.item()])

# pd.to_csv("result.csv",result)