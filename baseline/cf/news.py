
import math
import numpy as np
import pandas as pd
import os
import json
from econml.grf import CausalForest
import argparse
import torch


def curve(models,test_matrix, t_grid):
    n_test = t_grid.shape[1]
    t_grid_hat = np.zeros((3, n_test))
    t_grid_hat[0, :] = t_grid[0, :]
    n = test_matrix.shape[0]
    model1,model2 = models[0],models[1]

    for _ in range(n_test):
        t = np.full((n,), t_grid[0, _])
        x = test_matrix[:,1:-3]
        theta1,theta2 = model1.predict(x).squeeze(),model2.predict(x).squeeze()
        t_grid_hat[1, _] = (theta1*t).mean()
        t_grid_hat[2, _] = (theta2*t).mean()

    mse1 = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean()
    mse2 = ((t_grid_hat[2, :].squeeze() - t_grid[2, :].squeeze()) ** 2).mean()
    return t_grid_hat, mse1,mse2



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/news', help='dir of eval dataset')
    parser.add_argument('--data_split_dir', type=str, default='/root/test01/research/CausalCVR/dataset/news/eval', help='dir of data_utils split')
    parser.add_argument('--save_dir', type=str, default='logs/news/eval', help='dir to save result')
    
    # common
    parser.add_argument('--num_dataset', type=int, default=20, help='num of datasets to train')

    args = parser.parse_args()
    # data_utils
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Result = {'Causal Forest':[]}

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt').cpu().numpy()
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt').cpu().numpy()

    for _ in range(num_dataset):
        cur_save_path = save_path + '/' + str(_)
        if not os.path.exists(cur_save_path):
            os.makedirs(cur_save_path)

        idx_train = torch.load(args.data_split_dir + '/' + str(_) + '/idx_train.pt').cpu().numpy()
        idx_test = torch.load(args.data_split_dir + '/' + str(_) + '/idx_test.pt').cpu().numpy()
        train_matrix = data_matrix[idx_train, :]
        test_matrix = data_matrix[idx_test, :]
        t_grid = t_grid_all[:, idx_test]

        t = train_matrix[:, 0]
        x = train_matrix[:, 1:-3]
        y1 = train_matrix[:,-3]
        y2 = train_matrix[:,-1]
        idx = np.where(y1>0)

        model1 = CausalForest(inference=False,fit_intercept=False)
        model2 = CausalForest(inference=False,fit_intercept=False)
        model1.fit(x,t,y1)
        model2.fit(x[idx],t[idx],y2[idx])

        t_grid_hat, mse1, mse2= curve([model1,model2],test_matrix, t_grid)

        print(_,' current mse1: ', mse1,' mse2: ',mse2)
        Result['Causal Forest'].append([mse1,mse2])

        with open(save_path + '/result.json', 'w') as fp:
            json.dump(Result, fp)

