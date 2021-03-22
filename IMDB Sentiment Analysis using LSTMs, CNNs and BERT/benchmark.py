# Naive bayes and XGBoost with Tfidf vectorization used as benchmarks

from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from time import time


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def text_preprocess(sentence):
    """
    :param sentence: String containing a sentence to be preprocessed
    :return: a list of ordered/ ready to be processed words
    """
    # 1st step: lowercase
    sentence = sentence.lower()
    # 2nd step: substituting <br /><br /> with lbreak
    sentence = sentence.replace('<br /><br />', ' ')
    # 3rd step: Insert spaces between alphabetic / numbers and non alphabetic characters
    sentence = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", sentence)
    sentence = re.sub("[0-9]+", lambda ele: " " + ele[0] + " ", sentence)
    return sentence


if __name__ == '__main__':

    print('Loading train, dev and test data...')
    # loading train, dev and test data
    ls = read_IMDB('IMDB Dataset.csv')
    train_ls, dev_ls, test_ls = train_test_split(ls)
    print('Done.\n')

    # Extract features and labels
    print('Extracting features and labels...')
    X_train = np.array([text_preprocess(item[0]) for item in train_ls])
    X_dev = np.array([text_preprocess(item[0]) for item in dev_ls])
    X_test = np.array([text_preprocess(item[0]) for item in test_ls])
    y_train = np.array([item[1] for item in train_ls], dtype=np.long)
    y_dev = np.array([item[1] for item in dev_ls], dtype=np.long)
    y_test = np.array([item[1] for item in test_ls], dtype=np.long)
    print('Done.\n')

    # Vectorize using TfIdf or CountVectorizer
    print('Vectorizing features with Tfidf...')
    # vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, max_df=0.5, min_df=0.00, stop_words='english')
    print('Vectorizing features with CountVectorizer...') # Performs better
    vectorizer = CountVectorizer(ngram_range=(1,3), stop_words='english')

    # Provide a vocabulary to the vectorizer from the training set, then transform all sentences
    X_train = vectorizer.fit_transform(X_train)
    X_dev = vectorizer.transform(X_dev)
    X_test = vectorizer.transform(X_test)
    print('Done.\n')

    # Fit Naive Bayes model
    print('Fitting Multinomial Naive Bayes model...')
    t0 = time()
    model_1 = MultinomialNB() # Multinomial can fit data from sparse matrix, GaussianNB needs dense
    model_1.fit(X_train, y_train)
    print(f'\nTraining time: {round(time() - t0, 3)} s')

    # Compute accuracy
    t0 = time()
    score_train = model_1.score(X_train, y_train)
    print(f'Prediction time(train): {round(time() - t0, 3)} s')

    t0 = time()
    score_test = model_1.score(X_test, y_test)
    print(f'Prediction time(test): {round(time() - t0, 3)} s')
    print('\nTrain set score: ', score_train)
    print('Test set score: ', score_test)

    t0 = time()
    model_2 = GradientBoostingClassifier()
    print('Fitting Gradient Boosting model...')
    model_2.fit(X_train, y_train)
    print(f'\nTraining time: {round(time() - t0, 3)} s')

    # Compute accuracy
    t0 = time()
    score_train = model_2.score(X_train, y_train)
    print(f'Prediction time(train): {round(time() - t0, 3)} s')

    t0 = time()
    score_test = model_2.score(X_test, y_test)
    print(f'Prediction time(test): {round(time() - t0, 3)} s')
    print('\nTrain set score: ', score_train)
    print('Test set score: ', score_test)
