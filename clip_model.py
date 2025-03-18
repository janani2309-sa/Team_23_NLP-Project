# clip_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from data_processing import HatefulMemesDataset, transform, tokenizer
import pandas as pd

class CLIPHatefulMemeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    def forward(self, image, text):
        outputs = self.model(image, text)
        return outputs.logits_per_image

def train_clip(model, dataloader, epochs=5, lr=1e-4):
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
df = pd.read_json('hateful_memes.jsonl', lines=True)
dataset = HatefulMemesDataset(df, 'data/images', transform=transform, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train CLIP Model
clip_model = CLIPHatefulMemeModel()
train_clip(clip_model, dataloader)
