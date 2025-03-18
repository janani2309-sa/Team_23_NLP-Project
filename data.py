# data_processing.py - Data Loading, Cleaning, EDA, Visualization

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
from wordcloud import WordCloud
from torch.utils.data import Dataset, DataLoader
import os

# Load Data
def load_data(filepath):
    df = pd.read_json(filepath, lines=True)
    print("Data Loaded Successfully!")
    print(df.head())
    return df

# Data Cleaning
def clean_data(df):
    df.dropna(inplace=True)
    df = df[df['label'].isin([0, 1])]
    return df

# Data Visualization
def visualize_data(df):
    sns.countplot(x='label', data=df, palette='coolwarm')
    plt.title("Distribution of Labels")
    plt.show()
    
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(df['text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Dataset Class
class HatefulMemesDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, tokenizer=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['img']}.jpg")
        image = Image.open(img_path).convert("RGB")
        text = row['text']
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
        
        if self.tokenizer:
            text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        
        return image, text, torch.tensor(label, dtype=torch.long)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and Process Data
filepath = 'hateful_memes.jsonl'
df = load_data(filepath)
df = clean_data(df)
visualize_data(df)

# Initialize Dataset and DataLoader
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = HatefulMemesDataset(df, 'data/images', transform=transform, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Data Processing Complete!")
