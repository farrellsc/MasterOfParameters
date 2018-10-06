import tensorflow as tf
import numpy as np
import keras.backend as K


# The custom accuracy metric used for this task
def myAccuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s


# prints the results
def get_results(x, y_, revsere_decoder_index):
    # print("input     : " + str(x))
    # print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    return str(onehot_to_seq(y_, revsere_decoder_index).upper())


# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n=3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])
