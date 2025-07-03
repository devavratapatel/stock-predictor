import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv("C:/Users/Devavrata/proproject/labeled_data.csv")

tokenized = df["full_text"].apply(word_tokenize)
all_words = [word.lower() for tokens in tokenized for word in tokens]
vocab = Counter(all_words)
vocab = {word: i+2 for i, (word, freq) in enumerate(vocab.items()) if freq>= 2}
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_sentence(sentence, vocab, max_len=20):
    tokens = word_tokenize(sentence.lower())
    tokens = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))  
    return tokens

df["encoded"] = df["full_text"].apply(lambda x: encode_sentence(x, vocab))

class RedditDataset(Dataset):
    def __init__(self, df):
        self.x = list(df["encoded"])
        self.y = list(df["label"])
        self.log_upvotes = torch.tensor(df["log_upvotes"].values, dtype=torch.float32)
        self.sentiment = torch.tensor(df["sentiment"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),  
            torch.tensor(self.log_upvotes[idx], dtype=torch.float32), 
            torch.tensor(self.sentiment[idx], dtype=torch.float32), 
            torch.tensor(torch.tensor(self.y[idx], dtype=torch.float32)))
    
train_df, test_df = train_test_split(df, test_size = 0.2, stratify=df["label"])
train_dataset = RedditDataset(train_df)
test_dataset = RedditDataset(test_df)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

print("Label distribution:")
print(df['label'].value_counts(normalize=True))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)
        
        self.word_lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            dropout=0.3 if num_layers > 1 else 0,
            batch_first=True
        )
        
        
        self.word_attention = nn.Sequential(
            nn.Linear(2*hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        
        self.sent_lstm = nn.LSTM(
            2*hidden_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0,
            batch_first=True
        )
        
        
        self.sent_attention = nn.Sequential(
            nn.Linear(2*hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        
        self.feature_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 2*hidden_dim)
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(4*hidden_dim, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    for i in range(0, param.size(0), self.hidden_dim):
                        nn.init.xavier_uniform_(param[i:i+self.hidden_dim])
                elif 'embedding' not in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, lengths, upvotes, sentiment):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        word_out, _ = self.word_lstm(packed)
        word_out, _ = pad_packed_sequence(word_out, batch_first=True)
        
        word_attn = self.word_attention(word_out)
        word_attn = torch.transpose(word_attn, 1, 2)
        word_attn_vectors = torch.bmm(word_attn, word_out).squeeze(1)
        
        sent_out, _ = self.sent_lstm(word_out)
        
        sent_attn = self.sent_attention(sent_out)
        sent_attn = torch.transpose(sent_attn, 1, 2)
        sent_attn_vectors = torch.bmm(sent_attn, sent_out).squeeze(1)
        
        text_features = torch.cat([word_attn_vectors, sent_attn_vectors], dim=1)
        
        features = torch.stack([upvotes, sentiment], dim=1)
        processed_features = self.feature_net(features)
        
        combined = torch.cat([text_features, processed_features], dim=1)
        

        return self.classifier(combined).squeeze()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(
    vocab_size=max(vocab.values())+1, 
    embed_dim=128,
    hidden_dim=64
).to(device)


optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()


pos_weight = torch.tensor([(len(df)-df['label'].sum())/df['label'].sum()]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  

for epoch in range(100):
    model.train()
    total_loss = 0

    for x, upv, sent, y,  in tqdm(train_dataloader):
        x, y, upv, sent = x.to(device), y.to(device), upv.to(device), sent.to(device)
        optimizer.zero_grad()
        output = model(x,upv,sent)
        loss = loss_fn(output, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss: .4f}")
    if (epoch+1)%10==0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")


model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x,upv, sent,y in test_dataloader:
        x, upv, sent, y= x.to(device), upv.to(device), sent.to(device), y.to(device)
        output = model(x,upv,sent)
        preds = (output>0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc: .4f}")

print(f"Train size: {len(df)}")
print(f"Test size: {len(test_df)}")