from MasterOfParameters.data.dataLoader import dataLoader
from MasterOfParameters.models.lstm1 import lstm1


database = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfParameters/data/"
myDataLoader = dataLoader(database + 'train.csv', database + 'test.csv', 512)

predpath = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfParameters/predictions/lstm1.csv"
myModel = lstm1(myDataLoader, predpath)
myModel.train()
myModel.predict()
