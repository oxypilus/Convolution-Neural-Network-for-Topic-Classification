from model import Sequences, Conv_model
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import random_split
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball']
twenty_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
data = {"news": twenty_train.data, "labels": twenty_train.target}

dataset = Sequences(data)

def split_train_valid_test(corpus, valid_ratio = 0.1, test_ratio = 0.1):
  test_length = int(len(corpus) * test_ratio)
  valid_length = int(len(corpus) * valid_ratio)
  train_length = len(corpus) - valid_length - test_length
  return random_split (corpus, lengths = [train_length, valid_length, test_length])

train_dataset, valid_dataset, test_dataset = split_train_valid_test (dataset)

BATCH_SIZE = 500

train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE)
test_loader = DataLoader(test_dataset,batch_size = BATCH_SIZE)
valid_loader = DataLoader(valid_dataset,batch_size = BATCH_SIZE)

inp_size = len(dataset.token2idx)
model = Conv_model(inp_size,30,10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

train_losses, valid_losses = [], []
for epoch in range(25):
    model.train()
    total_loss, total = 0,0
    for inputs, target in train_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total += 1
    train_loss = total_loss/total

    if epoch % 1 == 0:
        model.eval()
        total_loss, total = 0,0
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                loss = criterion(output, target)
                total_loss += loss.item()
                total += 1
        valid_loss = total_loss/total

    if len(valid_losses) > 2 and all(valid_loss >= loss for loss in valid_losses[-3:]):
        print("Stopping early")
        break

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(
          f'epoch #{epoch + 1:3d}\ttrain_loss: {train_loss:.3f}\tvalid_loss: {valid_loss:.3f}\n',
    )


model.eval()
test_accuracy, n_examples = 0, 0
y_true, y_pred = [], []
input_type = 'bow'

with torch.no_grad():
    for inputs, target in test_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        probs = model(inputs)
        probs = probs.detach().cpu().numpy()
        predict = np.argmax (probs, axis=1)

        target = target.cpu().numpy()
        target = np.argmax (target, axis = 1)

        y_true.extend(predict)
        y_pred.extend(target)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(classification_report(y_true, y_pred))

epoch_ticks = range(epoch+1)
plt.plot(epoch_ticks, train_losses)
plt.plot(epoch_ticks, valid_losses)
plt.title("Losses")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.xticks(epoch_ticks)
plt.show()


