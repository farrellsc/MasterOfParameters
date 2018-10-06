from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.DCRNN import DCRNN


database = "/root/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512, 6)

predpath = "/root/MasterOfParameters/predictions/DCRNN.csv"
myModel = DCRNN(myDataLoader, predpath)
myModel.train()
myModel.predict()
