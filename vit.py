from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import numpy as np

df = pd.read_csv('data/AllinAll.csv')
# print(df.head())

# Access the first row of the "Input.image_url1" column
first_image_url = df['Input.image_url1'].iloc[0]

# Print the first image URL
print(first_image_url)

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get(first_image_url, stream=True).raw)
img = transform(img)[None,]
out = model(img)

clsidx = torch.argmax(out)
print(clsidx.item())
print(out)

# Get the attention map from the model's output
attention_map = out.squeeze().detach().numpy()

# Normalize the attention map to a range of 0-1
attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

print(attention_map)

print(attention_map.shape)