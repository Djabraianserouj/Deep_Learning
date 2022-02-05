import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

epoch_train = 50
learning_rate_train = 0.001
early_stop = 3
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data_csv = pd.read_csv("data.csv", sep = ";")
train_dataset, test_dataset = train_test_split(data_csv, test_size= 0.10, train_size=0.90)
# print("train_dataset is : ", train_dataset)
# print("test_dataset is : ", test_dataset)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
"DATALOADER IS ALREADY DONE IN TRAINER.PY"
train_dataset = ChallengeDataset(train_dataset, "train")
test_dataset = ChallengeDataset(test_dataset, "val")

# create an instance of our ResNet model
# TODO
res = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
loss_criterion = t.nn.functional.binary_cross_entropy
optimizer = t.optim.Adam(res.parameters(), lr = learning_rate_train)
trainer = Trainer(res, loss_criterion, optimizer, train_dataset, test_dataset, cuda = True, early_stopping_patience = early_stop)
# go, go, go... call fit on trainer
#TODO
res = trainer.fit(epoch_train)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

