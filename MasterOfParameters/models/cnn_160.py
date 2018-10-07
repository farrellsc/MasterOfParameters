from MasterOfParameters.models.model import model
from keras.models import Model, Input
from keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Bidirectional, Dropout, Conv1D, Add, Reshape
from MasterOfParameters.utility.utils import *


class cnn_160(model):
    def __init__(self, myDataLoader, predbase, epoch):
        super(cnn_160, self).__init__(myDataLoader, predbase, epoch)

    def build_model(self):
        input = Input(shape=(self.maxlen_seq,))

        # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
        x = Embedding(input_dim=self.n_words, output_dim=128, input_length=self.maxlen_seq)(input)

        # Defining a bidirectional LSTM using the embedded representation of the inputs
        x_conv1 = Conv1D(64, 3, padding='same', activation='relu')(x)
        x_conv2 = Conv1D(64, 7, padding='same', activation='relu')(x)
        x_conv3 = Conv1D(64, 11, padding='same', activation='relu')(x)
        x = Add()([x_conv1, x_conv2, x_conv3])

        # Defining a bidirectional LSTM using the embedded representation of the inputs
        x = Bidirectional(GRU(units=128, return_sequences=True, recurrent_dropout=0.5))(x)
        x = Bidirectional(GRU(units=128, return_sequences=True, recurrent_dropout=0.5))(x)

        # A dense layer to output from the LSTM's 64 units to the appropriate number of tags to be fed into the decoder
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Decoder
        y = Dense(self.n_tags, activation="softmax")(x)

        # Defining the model as a whole and printing the summary
        self.model = Model(input, y)
        self.model.summary()

        # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", myAccuracy])