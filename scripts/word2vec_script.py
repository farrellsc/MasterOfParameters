from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.DCRNN import DCRNN
from gensim.models import Word2Vec
import numpy as np


database = "/root/MasterOfParameters/data/"
predpath = "/root/MasterOfParameters/predictions/GRU2_mini.csv"
# wvSize = 100
# myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 6)

# myModel = DCRNN(myDataLoader, predpath, 5)
myModel = DCRNN(None, predpath, 5)
# myModel.train()
# myModel.predict()
