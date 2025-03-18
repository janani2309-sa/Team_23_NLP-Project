import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data and Preprocessing
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

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Exploratory Data Analysis
# df = pd.read_json('hateful_memes.jsonl', lines=True)
# plt.figure(figsize=(8,5))
# sns.countplot(x='label', data=df, palette='coolwarm')
# plt.title('Label Distribution')
# plt.show()

# Facebook AI Baseline Model
class FacebookAIBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 256)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(256 + 768, 2)
    
    def forward(self, image, text):
        img_features = self.resnet(image)
        text_features = self.bert(**text).pooler_output
        features = torch.cat((img_features, text_features), dim=1)
        output = self.fc(features)
        return output

# Training Function
def train(model, dataloader, epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, texts, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            texts = {key: val.to(device) for key, val in texts.items()}
            
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# Example Training Execution
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = HatefulMemesDataset(df, 'data/images', transform=transform, tokenizer=tokenizer)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# model = FacebookAIBaseline()
# train(model, dataloader)
