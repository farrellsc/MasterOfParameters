from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.lstm1 import lstm1


database = "/root/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 6)

predpath = "/root/MasterOfParameters/predictions/lstm1.csv"
myModel = lstm1(myDataLoader, predpath)
myModel.train()
myModel.predict()
