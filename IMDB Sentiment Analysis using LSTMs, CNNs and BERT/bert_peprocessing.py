from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import *
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from train_utils import *
import json
import os


def ls_to_array(ls):
    """
    :param ls: a list as created by ultils.read_file
    :return: np.array of sentences included in the list
    """
    ls_0 = [item[0] for item in ls]
    return np.array(ls_0)


def text_preprocessing(sentence):
    """
    :param sentence:  string
    :return: preprocessed string
    """
    # 1st step: lowercase
    sentence = sentence.lower()
    # 2nd step: substituting <br /><br /> with lbreak
    sentence = sentence.replace('<br /><br />', 'linebreak ')
    # 3rd step: Insert spaces between alphabetic / numbers and non alphabetic characters
    sentence = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", sentence)
    sentence = re.sub("[0-9]+", lambda ele: " " + ele[0] + " ", sentence)
    return sentence


# Create a function to tokenize a set of texts
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


def store_bert_tokenized_data_to_disk(input_ids, attention_masks, labels, output_filename):
    """Takes data as created by preprocessing_for_bert and stores in json readable format."""
    input_ids_list = input_ids.tolist()
    attention_masks_list = attention_masks.tolist()
    labels_list = labels.tolist()
    to_store = (input_ids_list, attention_masks_list, labels_list)
    with open(output_filename, 'w') as output:
        json.dump(to_store, output)


def read_bert_tokenized_data_from_disk(input_filename):
    """Reads data previously stored to disk and loads them in memory."""
    with open(input_filename, 'r') as input:
        data = json.load(input)
    input_ids_list = data[0]
    attention_masks_list = data[1]
    labels_list = data[2]
    input_ids = torch.tensor(input_ids_list)
    attention_masks = torch.tensor(attention_masks_list)
    labels = torch.tensor(labels_list)
    return input_ids, attention_masks, labels


# Training Procedure
def train_bert(train_data, dev_data, model, n_epochs=3):
    """Train a model for N epochs given data and hyper-params."""
    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )
    # Total number of training steps
    total_steps = len(train_data) * n_epochs  # n. batches x n. epochs
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch_bert(train_data, model.train(), optimizer)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        # Scheduler Step
        scheduler.step()

        # Run **validation**
        val_loss, val_acc = run_epoch_bert(dev_data, model.eval(), optimizer)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        # Save model
        torch.save(model.state_dict(), 'my_model.pt')  # Saves every epoch

    return val_acc


def run_epoch_bert(data, model, optimizer):
    """Train model for one pass of train data, and return loss, accuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        inputs, masks, labels = batch[0], batch[1], batch[2]

        # Get output predictions
        out = model(inputs, masks)  # Only for BERT

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        batch_accuracy = compute_accuracy(predictions, labels)
        batch_accuracies.append(batch_accuracy)
        print(batch_accuracy)

        # Compute loss
        loss = F.cross_entropy(out, labels)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


if __name__ == '__main__':

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