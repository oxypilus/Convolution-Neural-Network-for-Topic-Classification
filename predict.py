import torch
import numpy as np
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
def predict_sentiment(text):
    tokenizer = vectorizer.build_analyzer()
    token2idx = vectorizer.vocabulary_
    encode = lambda x: [token2idx[token] for token in tokenizer(x) if token in token2idx]
    pad = lambda x: x + (124 - len(x)) * [token2idx["<PAD>"]]
    
    model.eval()
    with torch.no_grad():
      inputs = torch.LongTensor(pad(encode(text)[:124]), device = device)
      inputs = torch.reshape(inputs, (1,124))
      probs = model(inputs).to(device)
      probs = probs.detach().cpu().numpy()
      predict = np.argmax(probs, axis=1)[0]
      
      categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball']

      answer = categories[predict]

      return answer
     
     
from sklearn.datasets import fetch_20newsgroups 
import pandas as pd 
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball']
twenty_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
data = {"news": twenty_train.data, "labels": twenty_train.target}
df = pd.DataFrame(data)

inp =  df.news[3]
print(predict_sentiment(inp))