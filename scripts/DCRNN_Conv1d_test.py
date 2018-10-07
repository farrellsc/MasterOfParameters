from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.cnn_160 import cnn_160


database = "/root/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 3)

predpath = "/root/MasterOfParameters/predictions/cnn_160.csv"
myModel = cnn_160(myDataLoader, predpath, 20)
myModel.train()
myModel.predict()
