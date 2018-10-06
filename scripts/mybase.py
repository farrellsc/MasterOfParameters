from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.lstm1 import lstm1


database = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512)
# for further use
# dataloader.train_input_data
# dataloader.train_target_data
# dataloader.test_input_data
# dataloader.X_train
# dataloader.X_val
# dataloader.y_train
# dataloader.y_val

# ----------------------------------------DATA---------------------------------------------
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
#
# maxlen_seq = 512

# Loading and converting the inputs to trigrams
# train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
# train_input_grams = seq2ngrams(train_input_seqs)

# Same for test
# test_input_seqs = test_df['input'].values.T
# test_input_grams = seq2ngrams(test_input_seqs)

# Initializing and defining the tokenizer encoders and decoders based on the train set
# tokenizer_encoder = Tokenizer()
# tokenizer_encoder.fit_on_texts(train_input_grams)
# tokenizer_decoder = Tokenizer(char_level = True)
# tokenizer_decoder.fit_on_texts(train_target_seqs)

# Using the tokenizer to encode and decode the sequences for use in training
# Inputs
# train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
# train_input_data = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')

# Targets
# train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
# train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
# train_target_data = to_categorical(train_target_data)

# Splitting the data for train and validation sets
# X_train, X_val, y_train, y_val = \
#     train_test_split(train_input_data, train_target_data, test_size = .1, random_state = 0)

# Use the same tokenizer defined on train for tokenization of test
# test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
# test_input_data = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')


predbase = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfParameters/data/"
myModel = lstm1(myDataLoader, predbase)
myModel.train()
myModel.predict()
# ----------------------------------------MODEL---------------------------------------------
# Computing the number of words and number of tags to be passed as parameters to the keras model
# n_words = len(tokenizer_encoder.word_index) + 1
# n_tags = len(tokenizer_decoder.word_index) + 1

# input = Input(shape = (maxlen_seq,))
#
# # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
# x = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
#
# # Defining a bidirectional LSTM using the embedded representation of the inputs
# x = Bidirectional(GRU(units = 64, return_sequences = True, recurrent_dropout = 0.1))(x)
#
# # A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
# y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)
#
# # Defining the model as a whole and printing the summary
# model = Model(input, y)
# model.summary()

# Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
# model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy", myAccuracy])

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 512)               0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 512, 128)          1225984   
_________________________________________________________________
bidirectional_1 (Bidirection (None, 512, 128)          98816     
_________________________________________________________________
time_distributed_1 (TimeDist (None, 512, 9)            1161      
=================================================================
Total params: 1,325,961
Trainable params: 1,325,961
Non-trainable params: 0

"""

# ---------------------------------------------TRAIN------------------------------------------------
# # Training the model on the training data and validating using the validation set
# model.fit(X_train, y_train, batch_size = 128, epochs = 1, validation_data = (X_val, y_val), verbose = 1)

# Defining the decoders so that we can
# revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
# revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}

# y_test_pred = model.predict(test_input_data[:])
# y_test_string = []
# for i in range(len(test_input_data)):
#     y_test_string.append(get_results(test_input_seqs[i], y_test_pred[i], revsere_decoder_index))
# test_df['expected'] = pd.Series(y_test_string)
# del test_df['len']
# del test_df['input']
# test_df.to_csv(__file__.split(".")[0]+"_pred.csv", sep=',', encoding='utf-8', index=False)