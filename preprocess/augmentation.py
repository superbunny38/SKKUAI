import torch
import torchvision
import PIL.Image as Image
from torchvision import transforms
import pandas as pd

df = pd.read_csv('./train/grade_labels.csv')
transform = transforms.RandomRotation((15,180),)
for img_name in df['imname']:
    img = Image.open(f'./trainset1/{img_name}')
    img = transform(img)
    img.save(f'./trainset5/{img_name}')