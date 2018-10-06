import pandas as pd
from MasterOfParameters.utility.utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class dataLoader:
    def __init__(self, train_path, test_path, maxlen_seq, ngrams):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.maxlen_seq = maxlen_seq
        self.ngrams = ngrams

        self.tokenizer_encoder = Tokenizer()
        self.tokenizer_decoder = Tokenizer(char_level=True)

        # for further use
        self.train_input_data = None
        self.train_target_data = None
        self.test_input_data = None
        self.test_input_seqs = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        self.preprocess()
        self.split_data()

    def preprocess(self):
        train_input_seqs, train_target_seqs = \
            self.train_df[['input', 'expected']][(self.train_df.len <= self.maxlen_seq)].values.T
        train_input_grams = seq2ngrams(train_input_seqs, self.ngrams)

        self.test_input_seqs = self.test_df['input'].values.T
        test_input_grams = seq2ngrams(self.test_input_seqs, self.ngrams)

        self.tokenizer_encoder.fit_on_texts(train_input_grams)
        self.tokenizer_decoder.fit_on_texts(train_target_seqs)

        train_input_data = self.tokenizer_encoder.texts_to_sequences(train_input_grams)
        self.train_input_data = sequence.pad_sequences(train_input_data, maxlen=self.maxlen_seq, padding='post')

        test_input_data = self.tokenizer_encoder.texts_to_sequences(test_input_grams)
        self.test_input_data = sequence.pad_sequences(test_input_data, maxlen=self.maxlen_seq, padding='post')

        train_target_data = self.tokenizer_decoder.texts_to_sequences(train_target_seqs)
        train_target_data = sequence.pad_sequences(train_target_data, maxlen=self.maxlen_seq, padding='post')
        self.train_target_data = to_categorical(train_target_data)

    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.train_input_data, self.train_target_data, test_size=.1, random_state=0)
