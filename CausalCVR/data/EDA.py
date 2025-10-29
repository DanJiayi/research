import numpy as np
import pandas as pd
import torch
load_path= '/root/test01/research/CausalCVR/dataset/simu2/eval'
data_train = pd.read_csv(load_path + '/' + '0/train.txt', header=None, sep=' ').to_numpy()
data_test = pd.read_csv(load_path + '/' + '0/test.txt', header=None, sep=' ').to_numpy()
data = np.vstack([data_train,data_test])
print(data.shape)
y1,y2 = data[:,-3],data[:,-1]
print('y1:',y1.mean(),y1.std())
print('y2:',y2.mean(),y2.std())
mask = (y1==1)
print('y2|y1:',y2[mask].mean(),y2[mask].std())

data_dir = '/root/test01/research/CausalCVR/dataset/news'
data_split_dir = '/root/test01/research/CausalCVR/dataset/news/eval'
idx_train = torch.load(data_split_dir + '/' +  '0/idx_train.pt')
idx_test = torch.load(data_split_dir + '/' + '0/idx_test.pt')
data_matrix = torch.load(data_dir + '/data_matrix.pt').cpu().numpy()
train_matrix = data_matrix[idx_train, :]
test_matrix = data_matrix[idx_test, :]
data = np.vstack([train_matrix ,test_matrix])
print(data.shape)
y1,y2 = data[:,-3],data[:,-1]
print('y1:',y1.mean(),y1.std())
print('y2:',y2.mean(),y2.std())
mask = (y1==1)
print('y2|y1:',y2[mask].mean(),y2[mask].std())