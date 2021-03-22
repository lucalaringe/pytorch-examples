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
from lstm import LSTMSentimentClassifier
from cnn import CNNSentimentClassifier

def main():
    # loading train, dev and test toy data
    # train_ls = read_file('senti_binary.train.txt')
    # dev_ls = read_file('senti_binary.dev.txt')
    # test_ls = read_file('senti_binary.test.txt')

    # Loading IMDB data if batches not already saved
    if not (os.path.exists('train_batches.json') and\
            os.path.exists('dev_batches.json') and\
            os.path.exists('test_batches.json')):
        print('Loading the data in memory...')
        ls = read_IMDB('IMDB Dataset.csv')
        train_ls, dev_ls, test_ls = train_test_split(ls)
        print('Done.\n')

    # Loading the vocabulary if in memory, otherwise create it
    if os.path.exists('my_voc.pkl'):
        print('Loading the vocabulary in memory...')
        with open('my_voc.pkl', 'rb') as input:
            my_voc = pickle.load(input)
            print('Done.\n')
    else:
        print('Creating and saving the vocabulary...')
        my_voc = WordVocabulary()
        my_voc.add_corpus_from_list(train_ls)
        # Save my_voc
        with open('my_voc.pkl', 'wb') as output:
            pickle.dump(my_voc, output, pickle.HIGHEST_PROTOCOL)
            print('Done.\n')

    # Creating Datasets and Batchify data
    batch_size = 32
    if os.path.exists('train_batches.json') and os.path.exists('dev_batches.json') and os.path.exists('test_batches.json'):
        print('Loading batches in memory...')
        train_batches = read_batches_from_disk('train_batches.json')
        dev_batches = read_batches_from_disk('dev_batches.json')
        test_batches = read_batches_from_disk('test_batches.json')
        print('Done\n')
    else:
        print('Instantiating, batchifying and saving the datasets...')
        # Instantiating Datasets
        train = MovieReviewsDataset(train_ls, my_voc)
        dev = MovieReviewsDataset(dev_ls, my_voc)
        test = MovieReviewsDataset(test_ls, my_voc)
        # Batchifying
        train_batches = batchify_data(train, batch_size)
        dev_batches = batchify_data(dev, batch_size)
        test_batches = batchify_data(test, batch_size)
        # Saving with json
        store_batches_to_disk(train_batches, 'train_batches.json')
        store_batches_to_disk(dev_batches, 'dev_batches.json')
        store_batches_to_disk(test_batches, 'test_batches.json')
        print('Done.\n')

    # Load model if already in memory, otherwise random initialization of the weights
    if os.path.exists('my_model.pt'):
        print('Loading the model...')
        # my_model = LSTMSentimentClassifier(my_voc, 50, 40, 20, 10)
        my_model = CNNSentimentClassifier(my_voc, max_len, 100, 6, 4, 3, 3)
        my_model.load_state_dict(torch.load('my_model.pt'))
        my_model.eval()
        print('Done.\n')
    else:
        print('Initializing the model...')
        # my_model = LSTMSentimentClassifier(my_voc, 50, 40, 20, 10)
        max_len = train_batches[0]['x'].shape[1]
        print(max_len)
        my_model = CNNSentimentClassifier(my_voc, max_len, 50, 6, 4, 3, 3)
        print('Done.\n')

    print('Starting training...\n')
    # Train the model
    train_model(train_batches, dev_batches, my_model, nesterov=True)

    # Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, my_model.eval(), None)
    print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle.
    np.random.seed(314)  # for reproducibility
    torch.manual_seed(314)
    main()