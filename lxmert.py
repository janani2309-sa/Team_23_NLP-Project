import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LxmertTokenizer, LxmertModel
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
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
        img_path = f"{self.img_dir}/{row['img']}.jpg"
        image = Image.open(img_path).convert("RGB")
        text = row['text']
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
        
        if self.tokenizer:
            text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        
        return image, text, torch.tensor(label, dtype=torch.long)

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data Loading and EDA
df = pd.read_json('hateful_memes.jsonl', lines=True)
print("Dataset Preview:")
print(df.head())

sns.countplot(x='label', data=df)
plt.title("Label Distribution")
plt.show()

# Tokenizer
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# Dataset and Dataloader
dataset = HatefulMemesDataset(df, 'data/images', transform=transform, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define LXMERT Model
class LXMERTMemeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.fc = nn.Linear(768, 2)
    
    def forward(self, image, text):
        outputs = self.lxmert(input_ids=text['input_ids'].squeeze(1), visual_feats=image)
        return self.fc(outputs.pooled_output)

# Training Function
def train(model, dataloader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
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

# Model Training
model = LXMERTMemeModel()
train(model, dataloader)
