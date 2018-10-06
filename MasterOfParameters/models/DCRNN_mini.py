from MasterOfParameters.models.model import model
from keras.models import Model, Input
from keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Bidirectional, Dropout
from MasterOfParameters.utility.utils import *


class DCRNN_mini(model):
    def __init__(self, myDataLoader, predbase, epoch):
        super(DCRNN_mini, self).__init__(myDataLoader, predbase, epoch)

    def build_model(self):
        input = Input(shape=(self.maxlen_seq,))

        # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
        x = Embedding(input_dim=self.n_words, output_dim=128, input_length=self.maxlen_seq)(input)

        # Defining a bidirectional LSTM using the embedded representation of the inputs
        x = Bidirectional(GRU(units=128, return_sequences=True, recurrent_dropout=0.5))(x)
        x = Bidirectional(GRU(units=128, return_sequences=True, recurrent_dropout=0.5))(x)
        x = Bidirectional(GRU(units=128, return_sequences=True, recurrent_dropout=0.5))(x)

        # A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        y = Dense(self.n_tags, activation="softmax")(x)

        # Defining the model as a whole and printing the summary
        self.model = Model(input, y)

        # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", myAccuracy])