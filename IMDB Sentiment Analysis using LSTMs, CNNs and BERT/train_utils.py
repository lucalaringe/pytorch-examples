# importing the libraries I need
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import *


class MovieReviewsDataset(Dataset):
    def __init__(self, ls, vocabulary):
        self._ls = ls
        self.vocabulary = vocabulary
        self.x = [vocabulary.sentence_to_idx(item[0]) for item in ls]
        self.x_length = [len(sentence) for sentence in self.x]
        self.y = [item[1] for item in ls]

    def __getitem__(self, idx):
        return self.x_padded[idx], self.y[idx]

    def __len__(self):
        return len(self._ls)

    def shuffle(self):
        permutation = np.array([i for i in range(len(self))])
        np.random.shuffle(permutation)
        self.x = [self.x[i] for i in permutation]
        self.x_length = [self.x_length[i] for i in permutation]
        self.y = [self.y[i] for i in permutation]


def batchify_data(dataset, batch_size=32):
    """Takes a dataset and groups it into batches."""
    dataset.shuffle()
    x_tensor = [torch.tensor(item) for item in dataset.x]
    x_padded = pad_sequence(x_tensor, batch_first=True, padding_value=dataset.vocabulary.PAD_INDEX)

    # Only take batch_size chunks (i.e. drop the remainder)
    N = (len(dataset) // batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': x_padded[i:i + batch_size],
            'len_x': dataset.x_length[i:i + batch_size],
            'y': torch.tensor(dataset.y[i:i + batch_size],
                              dtype=torch.long)})  # .reshape(batch_size, -1) (we want 1d targets)
    return batches


def store_batches_to_disk(batches, output_file_name):
    """Takes batches as created by batchify_data and stores in json readable format."""
    batches_readable = copy.deepcopy(batches)
    for batch in batches_readable:
        batch['x'] = batch['x'].tolist()
        batch['y'] = batch['y'].tolist()

    with open(output_file_name, 'w') as output:
        json.dump(batches_readable, output)


def read_batches_from_disk(input_file_name):
    """Reads batches previously stored to disk and loads them in memory."""
    with open(input_file_name, 'r') as input:
        batches = json.load(input)
    for batch in batches:
        batch['x'] = torch.tensor(batch['x'])
        batch['y'] = torch.tensor(batch['y'])
    return batches


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""

    def forward(self, input):
        return input.view(input.size(0), -1)


# Training Procedure
def train_model(train_data, dev_data, model, initial_lr=0.1, gamma=0.95, momentum=0.9, nesterov=False, n_epochs=10):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, nesterov=nesterov)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        # Scheduler Step
        scheduler.step()

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        # Save model
        torch.save(model.state_dict(), 'my_model.pt')  # Saves every epoch

    return val_acc


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, accuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x, len_x and y
        x, len_x, y = batch['x'], batch['len_x'], batch['y']

        # Get output predictions
        # out = model(x, len_x) # For LSTM
        out = model(x)  # For CNN

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, y))

        # Compute loss
        loss = F.cross_entropy(out, y)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


if __name__ == '__main__':
    # loading train, dev and test data
    train_ls = read_file('senti_binary.train')
    dev_ls = read_file('senti_binary.dev')
    # test_ls = read_file('senti_binary.test')

    my_dict = WordVocabulary()
    my_dict.add_corpus_from_list(dev_ls)

    train = MovieReviewsDataset(train_ls, my_dict)
    batches = batchify_data(train, 32)
    print(batches)
