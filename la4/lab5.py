import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report
import pandas as pd

train_data = np.load('datal/training_sentences.npy',allow_pickle=True)
train_labels = np.load('datal/training_labels.npy',allow_pickle=True)

test_data = np.load('datal/test_sentences.npy',allow_pickle=True)
test_labels = np.load('datal/test_labels.npy',allow_pickle=True)


def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')
    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data) 
        return (scaled_train_data, scaled_test_data)
    else:
        return (train_data, test_data)


class BagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.words = []

    def build_vocabulary(self, data):
        for message in data:
            for word in message:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
                    self.words.append(word)
    def get_features(self, data):
        num_samples = len(data)
        a = np.zeros((num_samples,len(self.vocabulary)))
        for i,mes in enumerate(data):
            for word in mes:
                if word in self.vocabulary:
                    ct = self.vocabulary[word]
                    a[i,ct] += i
        return a


bow = BagOfWords()
bow.build_vocabulary(train_data)
train_features = bow.get_features(train_data)
test_features = bow.get_features(test_data)
train_features,test_features = normalize_data(train_features,test_features,'l2')


svm_model = svm.SVC(C=1,kernel='linear')
svm_model.fit(train_features,train_labels)
predict = svm_model.predict(test_features)
no_corect_predictions = np.sum(predict == test_labels)
accuracy = no_corect_predictions / len(predict)
print(accuracy)
print(f1_score(test_labels,predict))


coefficients = svm_model.coef_[0]

# Obținem vocabularul (cuvintele) de la obiectul Bag of Words
vocabulary = bow.words

# Creăm un DataFrame pentru a asocia fiecare cuvânt cu coeficientul său
words_df = pd.DataFrame({'word': vocabulary, 'coefficient': coefficients})

# Sortăm DataFrame-ul în funcție de coeficienți
words_df = words_df.sort_values(by='coefficient')

# Afișăm cele mai negative (spam) 10 cuvinte
print(words_df.head(10))

# Afișăm cele mai pozitive (non-spam) 10 cuvinte
print(words_df.tail(10))