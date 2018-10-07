from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.DCRNN import DCRNN
from gensim.models import Word2Vec
import numpy as np


database = "/root/MasterOfParameters/data/"
predpath = "/root/MasterOfParameters/predictions/GRU2_mini.csv"
wvSize = 100
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 6)

print("---------------------------loading data----------------------------------")
train_input_data_str = [[str(one) for one in two] for two in myDataLoader.train_input_data]
test_input_data_str = [[str(one) for one in two] for two in myDataLoader.test_input_data]
wvModel = Word2Vec(train_input_data_str, size=wvSize, window=5, min_count=1, workers=4)

new_train_input = np.zeros([len(train_input_data_str), len(train_input_data_str[0]), wvSize])     # 4960*512*100
for i in range(new_train_input .shape[0]):
    for j in range(new_train_input .shape[1]):
        new_train_input[i, j] = wvModel.wv[train_input_data_str[i][j]]
new_train_input = new_train_input.reshape([new_train_input.shape[0], 1, new_train_input.shape[2], new_train_input[3]])

new_test_input = np.zeros([len(test_input_data_str), len(test_input_data_str[0]), wvSize])  # 119*512*100
for i in range(new_test_input.shape[0]):
    for j in range(new_test_input.shape[1]):
        new_test_input[i, j] = wvModel.wv[test_input_data_str[i][j]]
new_test_input = new_test_input.reshape([new_test_input.shape[0], 1, new_test_input.shape[2], new_test_input[3]])

myDataLoader.train_input_data = new_train_input
myDataLoader.test_input_data = new_test_input
myDataLoader.split_data()


print("---------------------------training----------------------------------")
myModel = DCRNN(myDataLoader, predpath, 5)
myModel.train()
myModel.predict()
