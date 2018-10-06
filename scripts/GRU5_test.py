from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.GRU5 import GRU5


database = "/root/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 6)

predpath = "/root/MasterOfParameters/predictions/GRU5.csv"
myModel = GRU5(myDataLoader, predpath)
myModel.train()
myModel.predict()
