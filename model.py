import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sequences(Dataset):
    def __init__(self, data):
        max_seq_len = 124
        df = pd.DataFrame(data)
        vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        vectorizer.fit(df.news.tolist())
        onehotencoder = OneHotEncoder()
        targets = np.array(df.labels.tolist())
        targets = onehotencoder.fit_transform(targets.reshape(-1,1))

        tokenizer = vectorizer.build_analyzer()
        token2idx = vectorizer.vocabulary_
        token2idx["<PAD>"] = len(token2idx)

        self.encode = lambda x: [token2idx[token] for token in tokenizer(x) if token in token2idx]
        self.pad = lambda x: x + (max_seq_len - len(x)) * [token2idx["<PAD>"]]

        sequences = [self.encode(sequence)[:max_seq_len] for sequence in df.news.tolist()]
        sequences, self.labels = zip(*[(sequence, label) for sequence, label in zip(sequences, targets.toarray()) if sequences])
        self.sequences = [self.pad(sequence) for sequence in sequences]

        self.sequences = torch.LongTensor(self.sequences)#.toarray())
        self.labels = torch.FloatTensor(targets.toarray())
        self.token2idx = vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        with open("vectorizer.pkl", "wb") as f:
          pickle.dump(vectorizer, f)

    def __getitem__(self, i):
        return self.sequences[i, :], self.labels[i]

    def __len__(self):
        return self.sequences.shape[0]

class Conv_model(nn.Module):
    def __init__ (self, inp_size, hidden1, hidden2):
        super(Conv_model, self).__init__()

        self.filter_sizes = [3, 4, 5, 6]
        self.num_filters = [15, 15, 15, 15]
        self.dropout = nn.Dropout(p=0.5)

        self.embed = nn.Embedding(inp_size, 300, inp_size - 1)
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels = 300, out_channels = self.num_filters[i], kernel_size = self.filter_sizes[i]) for i in range(len(self.filter_sizes))])
        self.fc1 = nn.Linear(60,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)

    def forward (self, inp):
        x = self.embed(inp)
        x_reshaped = x.permute(0,2,1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                for x_conv in x_conv_list]
        out = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                            dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


