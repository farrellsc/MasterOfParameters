from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.GRU2 import GRU2


database = "/root/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 6)

predpath = "/root/MasterOfParameters/predictions/GRU2_mini.csv"
myModel = GRU2(myDataLoader, predpath)
myModel.train()
myModel.predict()
