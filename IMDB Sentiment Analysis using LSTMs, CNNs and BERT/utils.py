# importing the libraries I need
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from utils import *


# Defining a file reader to IMDB data
def read_IMDB(file_name):
    """
    :param file_name: Takes as input the IMDB dataset file name as a string
    :return: and outputs a list of lists containg datapoins consisting of a sentence (string) and a target (an integer)
    """
    with open(file_name, 'r', encoding="latin-1") as text_file:
        l = []
        target_string_to_int = {'negative': 0, 'positive': 1}
        lines = text_file.readlines()[1:]
        for i, line in enumerate(lines):
            l.append(line.rstrip().rsplit(',', 1))

        # Transforming target variables from stings to integers
        l = [(sentence[1:-1], target_string_to_int[target]) for sentence, target in l]
        return l


def train_test_split(ls, train=0.4, dev=0.1, test=0.5):
    """
    :param ls: list of 2-dimensional tuples. First element is a sentence, second is a prediction (0/1)
    :return: 3 lists for train test split. train 40%, dev 10%, test 50% defaults
    """
    assert sum((train, dev, test)) == 1, 'train, dev and test must sum to 1.'
    n = len(ls)
    train_ls = ls[0:int(train * n)]
    dev_ls = ls[int(train * n): int((train + dev) * n)]
    test_ls = ls[int((train + dev) * n):n + 1]
    return train_ls, dev_ls, test_ls


# Defining a file reader to read train, dev and test toy data (used during development of model architecture)
def read_file(file_name):
    """
    :param file_name: Takes as input either the train, dev or test file names as a string
    :return: and outputs a list of lists containg datapoints consisting of a sentence (string) and a target (an integer)
    """
    with open(file_name, 'r') as text_file:
        l = []
        for line in text_file:
            l.append(line.rstrip().split('\t'))

        # Transforming target variables from stings to integers
        l = [(sentence.encode('latin-1').decode('utf-8'), int(target)) for sentence, target in l]
        return l


# Let's now build our dictionary of words
class WordVocabulary:
    # Class Attributes
    _PAD_TOKEN = '<pad>'
    _UNK_TOKEN = '__unknown__'
    _BEG_TOKEN = '<beg>'
    _END_TOKEN = '<end>'
    PAD_INDEX = 0
    _UNK_INDEX = 1
    _BEG_INDEX = 2
    _END_INDEX = 3

    stopwords = set(stopwords.words('english'))
    _apostrophe_dict = {
        "n't": "not",
        "'d": "would",
        "'ll": "will",
        "'m": "am",
        "'s": "is",
        "'re": "are",
        "'ve": "have",
        "wo": "will",  # won't -> [wo, n't] -> [will, not]
        "sha": "shall"  # shan't -> [sha, n't] -> [shall, not]
    }

    _lemmatizer = WordNetLemmatizer()

    def __init__(self):
        self._length = 4
        self.idx_to_word = {self.PAD_INDEX: self._PAD_TOKEN, self._UNK_INDEX: self._UNK_TOKEN, \
                            self._BEG_INDEX: self._BEG_TOKEN, self._END_INDEX: self._END_TOKEN}
        self.word_to_idx = {self.idx_to_word[idx]: idx for idx in self.idx_to_word}

    def __len__(self):
        return self._length

    @staticmethod
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    @classmethod
    def text_preprocess(cls, sentence):
        """
        :param sentence: String containing a sentence to be preprocessed
        :return: a list of ordered/ ready to be processed words
        """
        # 1st step: lowercase
        sentence = sentence.lower()
        # 2nd step: substituting <br /><br /> with lbreak
        sentence = sentence.replace('<br /><br />', 'linebreak ')
        # 3rd step: Insert spaces between alphabetic / numbers and non alphabetic characters
        sentence = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", sentence)
        sentence = re.sub("[0-9]+", lambda ele: " " + ele[0] + " ", sentence)
        # 4th step: tokenization
        tokenized_sentence = word_tokenize(sentence)
        # # 5th step: remove stopwords
        # tokenized_sentence = [word for word in tokenized_sentence if word not in cls.stopwords]
        # 6th step: apostrophe handling
        tokenized_sentence = [cls._apostrophe_dict[word] if word in cls._apostrophe_dict else word for word in
                              tokenized_sentence]
        # 7th step: lemmatizing with the appropriate POS tag
        tokenized_sentence = [cls._lemmatizer.lemmatize(word, cls.get_wordnet_pos("was")) for word in tokenized_sentence]
        return [cls._BEG_TOKEN] + tokenized_sentence + [cls._END_TOKEN]

    def add_corpus(self, corpus):
        """
        :param corpus: a list of words
        :return: adds all words never seen the dictionary
        """
        unique_words = set(corpus)
        unique_words_never_seen = [word for word in unique_words if word not in self.word_to_idx]
        unique_words_never_seen.sort()  # In alphabetical order
        n = len(unique_words_never_seen)
        new_dict_idx_to_word = {(self._length + i): word for i, word in enumerate(unique_words_never_seen)}
        new_dict_word_to_idx = {new_dict_idx_to_word[idx]: idx for idx in new_dict_idx_to_word}
        self.idx_to_word.update(new_dict_idx_to_word)
        self.word_to_idx.update(new_dict_word_to_idx)
        self._length += n

    def add_corpus_from_list(self, ls):
        """
        :param ls: a list for which each element is a tuple whose first element is a string representing a sentence
        :return: adds all words never seen the dictionary
        """
        sentence_ls = [self.text_preprocess(datapoint[0]) for datapoint in ls]
        word_ls = [word for sentence in sentence_ls for word in sentence]
        self.add_corpus(word_ls)

    def sentence_to_idx(self, sentence):
        """
        :param sentence: String containing a sentence
        :return: list of idxs corresponding to the position in the vocabulary
        """
        tokenized_sentence = self.text_preprocess(sentence)
        tokenized_sentence_unk = [word if word in self.word_to_idx else self._UNK_TOKEN for word in tokenized_sentence]
        return [self.word_to_idx[word] for word in tokenized_sentence_unk]

    def sentence_to_idx_tensor(self, sentence):
        """
        :param sentence: String containing a sentence
        :return: tensor of idxs corresponding to the position in the vocabulary of size (1, length of the sentence)
        """
        return torch.tensor(self.sentence_to_idx(sentence)).reshape(1, -1)


class MovieReviewsDataset(Dataset):
    def __init__(self, ls, vocabulary):
        self._ls = ls
        self._vocabulary = vocabulary
        self.x = [torch.tensor(vocabulary.sentence_to_idx(item[0])) for item in ls]
        self.x_padded = pad_sequence(self.x, batch_first=True, padding_value=vocabulary.PAD_INDEX)
        self.y = [item[1] for item in ls]
        self._len = len(self._ls)

    def __getitem__(self, idx):
        return self.x_padded[idx], self.y[idx]

    def __len__(self):
        return self._len


if __name__ == '__main__':
    # loading train, dev and test data
    train_ls = read_file('senti_binary.train.txt')
    dev_ls = read_file('senti_binary.dev.txt')
    test_ls = read_file('senti_binary.test.txt')

    my_dict = WordVocabulary()
    my_dict.add_corpus_from_list(dev_ls)

    train = MovieReviewsDataset(train_ls, my_dict)
    print(train[1])
