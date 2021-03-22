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

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocabulary, embedding_dim, hidden_dim_1, hidden_dim_2, linear_hidden_dim):
        super().__init__()
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.linear_hidden_dim = linear_hidden_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.lstm_layer_1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim_1, batch_first=True,\
                                  bidirectional=True)
        self.lstm_layer_2 = nn.LSTM(input_size=self.hidden_dim_1*2, hidden_size=self.hidden_dim_2, batch_first=True,\
                                  bidirectional=True)
        self.attention = torch.rand(self.hidden_dim_2, requires_grad=True)
        self.dropout_layer_1 = nn.Dropout(0.2)
        self.linear_hidden_layer = nn.Linear(self.hidden_dim_2, self.linear_hidden_dim)
        self.a = nn.LeakyReLU()
        self.dropout_layer_2 = nn.Dropout(0.2)
        self.linear_layer = nn.Linear(self.linear_hidden_dim, 2)

    def forward(self, idx_sentence, len_sentence):
        embeddings = self.embedding(idx_sentence)
        embeddings_packed = pack_padded_sequence(embeddings, len_sentence,\
                                                 batch_first=True, enforce_sorted=False)
        out_pack_1, (ht_1, ct_1) = self.lstm_layer_1(embeddings_packed)
        out_pack_2, (ht_2, ct_2) = self.lstm_layer_2(out_pack_1)
        out_2_unpacked, out_2_lengths = pad_packed_sequence(out_pack_2)
        # Sum forward and backward h[t]
        out_2_unpacked_forward, out_2_unpacked_backward = torch.split(out_2_unpacked, self.hidden_dim_2, dim=2)
        out_2_unpacked_sum = out_2_unpacked_forward + out_2_unpacked_backward
        # Focus on the most important processed outputs in sequence
        attention_weights = F.softmax(torch.mul(out_2_unpacked_sum, self.attention).sum(dim=2), dim=1)
        attention_weights_casted = attention_weights.unsqueeze(2).repeat(1, 1, out_2_unpacked_sum.shape[2])
        weighted_out_2 = torch.mul(out_2_unpacked_sum, attention_weights_casted).sum(dim=0)
        # Dropout and Fully Connected Layers
        hidden_1_dropout = self.dropout_layer_1(weighted_out_2)
        hidden_2 = self.linear_hidden_layer(hidden_1_dropout)
        hidden_2_a = self.a(hidden_2)
        hidden_2_dropout = self.dropout_layer_2(hidden_2_a)
        output = self.linear_layer(hidden_2_dropout)
        return output

    def predict_prob(self, review):
        """
        :param review: string (theoretically a film review)
        :return: tensor representing the probability distribution over 0 (negative review) and 1 (positive review)
        """
        idx_tensor = self.vocabulary.sentence_to_idx_tensor(review)
        len_sentence = [idx_tensor.shape[1]]
        prob_distribution = F.softmax(self(idx_tensor, len_sentence), dim=1)
        return prob_distribution

    def predict(self, review):
        """
        :param review: string (theoretically a film review)
        :return: integer representing the predicted class
        """
        prob_distribution = self.predict_prob(review)
        return int(torch.argmax(prob_distribution, dim=1))