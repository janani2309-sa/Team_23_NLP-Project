
# resnet_text_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from transformers import BertModel
from torch.utils.data import DataLoader
from data_processing import HatefulMemesDataset, transform, tokenizer
import pandas as pd

class ResNetTextModel(nn.Module):
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

def train_resnet_text(model, dataloader, epochs=5, lr=1e-4):
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

# Load dataset
dataset = HatefulMemesDataset(df, 'data/images', transform=transform, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train ResNet + Text Model
resnet_text_model = ResNetTextModel()
train_resnet_text(resnet_text_model, dataloader)
