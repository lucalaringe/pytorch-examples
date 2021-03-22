import torch
import pickle
from main import LSTMSentimentClassifier
from utils import WordVocabulary
from cnn import CNNSentimentClassifier

# Retrieve Vocabulary
with open('my_voc.pkl', 'rb') as input:
    my_voc = pickle.load(input)

# Retrieve Model
# model = LSTMSentimentClassifier(my_voc, 50, 40, 20, 10)
model = CNNSentimentClassifier(my_voc, max_len)
model.load_state_dict(torch.load('my_model.pt'))
model.eval()

# Compute predictions
print(model.predict("amazing"))