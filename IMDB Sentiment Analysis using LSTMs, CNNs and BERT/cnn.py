# importing the libraries I need
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import nltk  # natural language toolkit
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from utils import *
from train_utils import *


class CNNSentimentClassifier(nn.Module):
    def __init__(self, vocabulary, max_len_sentence, embedding_dim=100, conv_1_window=4, pool_1_window=3, \
                 conv_2_window=4, pool_2_window=3, conv_1_channels=32, conv_2_channels=64):
        super().__init__()
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.embedding_dim = embedding_dim
        self.max_len_sentence = max_len_sentence
        self.conv_1_window = conv_1_window
        self.pool_1_window = pool_1_window
        self.conv_2_window = conv_2_window
        self.pool_2_window = pool_2_window
        self.conv_1_channels = conv_1_channels
        self.conv_2_channels = conv_2_channels
        self.flatten_dim = ((((self.max_len_sentence - self.conv_1_window + 1) // self.pool_1_window) - \
                            self.conv_2_window + 1) // self.pool_2_window)*self.conv_2_channels
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.conv_1 = nn.Conv2d(1, self.conv_1_channels, (self.conv_1_window, self.embedding_dim))
        self.a = nn.LeakyReLU()
        self.pool_1 = nn.MaxPool2d((self.pool_1_window, 1))
        self.conv_2 = nn.Conv2d(self.conv_1_channels , self.conv_2_channels, (self.conv_2_window, 1))
        self.pool_2 = nn.MaxPool2d((self.pool_2_window, 1))
        self.flatten = Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.dense_to_hidden = nn.Linear(self.flatten_dim, 32)
        self.dense_to_output = nn.Linear(32, 2)

    def forward(self, idx_sentence):
        padded_tensor = torch.zeros([idx_sentence.shape[0], self.max_len_sentence], dtype=torch.long)  # Zero Padding
        padded_tensor[0:idx_sentence.shape[0], 0:idx_sentence.shape[1]] = idx_sentence
        embeddings = self.embedding(padded_tensor)
        # Adding the channel Dimension
        embeddings_unsqueezed = embeddings.unsqueeze(1)
        conv_1 = self.conv_1(embeddings_unsqueezed)
        conv_1_a = self.a(conv_1)
        pool_1 = self.pool_1(conv_1_a)
        conv_2 = self.conv_2(pool_1)
        conv_2_a = self.a(conv_2)
        pool_2 = self.pool_2(conv_2_a)
        flat_layer = self.flatten(pool_2)
        flat_layer_dropout = self.dropout(flat_layer)
        hidden = self.dense_to_hidden(flat_layer_dropout)
        hidden_a = self.a(hidden)
        output = self.dense_to_output(hidden_a)
        return output

    def predict_prob(self, review):
        """
        :param review: string (theoretically a film review)
        :return: tensor representing the probability distribution over 0 (negative review) and 1 (positive review)
        """
        padded_tensor = torch.zeros([1, self.max_len_sentence], dtype=torch.long)  # Zero Padding
        idx_tensor = self.vocabulary.sentence_to_idx_tensor(review)
        padded_tensor[0, 0:idx_tensor.shape[1]] = idx_tensor
        prob_distribution = F.softmax(self(padded_tensor), dim=1)
        return prob_distribution

    def predict(self, review):
        """
        :param review: string (theoretically a film review)
        :return: integer representing the predicted class
        """
        prob_distribution = self.predict_prob(review)
        return int(torch.argmax(prob_distribution, dim=1))


if __name__ == '__main__':

    with open('my_voc.pkl', 'rb') as input:
        my_voc = pickle.load(input)

    s = torch.tensor([[i for i in range(1000)]])
    my_dict = WordVocabulary()
    m = CNNSentimentClassifier(my_voc, 1000)
    print(m.predict('fkjgdhsfd'))