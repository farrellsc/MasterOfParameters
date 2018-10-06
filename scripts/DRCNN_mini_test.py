from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.DCRNN_mini import DCRNN_mini


database = "/root/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512)

predpath = "/root/MasterOfParameters/predictions/DCRNN_mini.csv"
myModel = DCRNN_mini(myDataLoader, predpath)
myModel.train()
myModel.predict()
