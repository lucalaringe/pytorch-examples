import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from bert_peprocessing import *
import torch.nn.functional as F
from utils import *
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import os
import random


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for IMDB Classification Task.
    """

    def __init__(self, freeze_bert=True):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 64, 2  # 768 is the default dimension of the last hidden state returned by bert

        # Instantiate BERT model
        # BERT Base: 12 layers(transformer blocks), 12 attention heads, and 110 million parameters
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model (besides last transformer layer)
        if freeze_bert:
            modules = [self.bert.embeddings, *self.bert.encoder.layer[:9]]  # Freeze only first 9 transformer layers
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        # Dimensions are: 1) Batches 2) Tokens (1st token is [CLS]) 3) Neurons

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

    @staticmethod  # Copied from bert_preprocessing to add a predict method
    def preprocessing_for_bert(data, max_len=512, truncation=True):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data: np.array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # For every sentence...
        for sentence in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sentence = tokenizer.encode_plus(
                text=text_preprocessing(sentence),  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max_len,  # Max length to truncate/pad
                padding='max_length',  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                truncation=truncation,
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sentence.get('input_ids'))
            attention_masks.append(encoded_sentence.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks

    def predict_prob(self, review):
        """
        :param review: string (theoretically a film review)
        :return: tensor representing the probability distribution over 0 (negative review) and 1 (positive review)
        """
        input, mask = self.preprocessing_for_bert([review])
        prob_distribution = F.softmax(self(input, mask), dim=1)
        return prob_distribution

    def predict(self, review):
        """
        :param review: string (theoretically a film review)
        :return: integer representing the predicted class
        """
        prob_distribution = self.predict_prob(review)
        return int(torch.argmax(prob_distribution, dim=1))


def set_seed(seed_value=314):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


if __name__ == '__main__':

    # For reproducibility
    set_seed()

    if not (os.path.exists('train_bert.json') and os.path.exists('dev_bert.json') and os.path.exists('test_bert.json')):
        # loading train, dev and test data
        ls = read_IMDB('IMDB Dataset.csv')
        train_ls, dev_ls, test_ls = train_test_split(ls)
        # Extract features and labels
        X_train = np.array([item[0] for item in train_ls])
        X_dev = np.array([item[0] for item in dev_ls])
        X_test = np.array([item[0] for item in test_ls])
        y_train = np.array([item[1] for item in train_ls])
        y_dev = np.array([item[1] for item in dev_ls])
        y_test = np.array([item[1] for item in test_ls])

    # Bert can handle only up to 512 tokens so it will truncate the sequence automatically

    # Run function `preprocessing_for_bert` on the train set, validation and test set
    # Tokenize data if not done yet, or load them in memory.
    if os.path.exists('train_bert.json') and os.path.exists('dev_bert.json') and os.path.exists('test_bert.json'):
        print('Loading tokenized data in memory...')
        train_inputs, train_masks, train_labels = read_bert_tokenized_data_from_disk('train_bert.json')
        dev_inputs, dev_masks, dev_labels = read_bert_tokenized_data_from_disk('dev_bert.json')
        test_inputs, test_masks, test_labels = read_bert_tokenized_data_from_disk('test_bert.json')

    else:
        print('Tokenizing and saving data...')
        train_inputs, train_masks = preprocessing_for_bert(X_train)
        dev_inputs, dev_masks = preprocessing_for_bert(X_dev)
        test_inputs, test_masks = preprocessing_for_bert(X_test)
        train_labels = torch.tensor(y_train)
        dev_labels = torch.tensor(y_dev)
        test_labels = torch.tensor(y_test)
        store_bert_tokenized_data_to_disk(train_inputs, train_masks, train_labels, 'train_bert.json')
        store_bert_tokenized_data_to_disk(dev_inputs, dev_masks, dev_labels, 'dev_bert.json')
        store_bert_tokenized_data_to_disk(test_inputs, test_masks, test_labels, 'test_bert.json')

    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = 32

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    # Create the DataLoader for our test set
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Load model if already in memory, otherwise random initialization of the weights
    if os.path.exists('my_model.pt'):
        print('Loading the model...')
        my_model = BertClassifier(freeze_bert=True)
        my_model.load_state_dict(torch.load('my_model.pt'))
        my_model.eval()
        print('Done.\n')
    else:
        print('Initializing the model...')
        my_model = BertClassifier(freeze_bert=True)
        print('Done.\n')

    # Train the model
    train_bert(train_dataloader, dev_dataloader, my_model)

    # Evaluate the model on test data
    loss, accuracy = run_epoch_bert(test_dataloader, my_model.eval(), None)
    print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))
