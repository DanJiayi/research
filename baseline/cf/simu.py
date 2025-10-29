
# import torch
import math
import numpy as np
import pandas as pd
import os
import json
from econml.grf import CausalForest
import argparse

def predict_value(model, X, T):
    """
    Reconstructs E[Y | X, T] from CausalForest local linear parameters.
    X : (n, d) covariates
    T : (n,) or scalar treatment values
    """
    theta = model.predict_full(X)   # (n, 2): intercept and slope
    theta0 = theta[:, 1]
    theta1 = theta[:, 0]
    T = np.asarray(T).reshape(-1)
    if T.shape[0] == 1:  # broadcast scalar T
        T = np.full(X.shape[0], T.item())
    return theta0 + theta1 * T


def curve(models,test_matrix, t_grid):
    n_test = t_grid.shape[1]
    t_grid_hat = np.zeros((3, n_test))
    t_grid_hat[0, :] = t_grid[0, :]
    n = test_matrix.shape[0]
    model1,model2 = models[0],models[1]

    for _ in range(n_test):
        t = np.full((n,), t_grid[0, _])
        x = test_matrix[:,1:-3]
        # theta1,theta2 = model1.predict(x).squeeze(),model2.predict(x).squeeze()
        # t_grid_hat[1, _] = (theta1*t).mean()
        # t_grid_hat[2, _] = (theta2*t).mean()

        t_grid_hat[1, _] = (predict_value(model1, x, t)).mean()
        t_grid_hat[2, _] = (predict_value(model2, x, t)).mean()

    mse1 = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean()
    mse2 = ((t_grid_hat[2, :].squeeze() - t_grid[2, :].squeeze()) ** 2).mean()
    return t_grid_hat, mse1,mse2



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/simu2/eval', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=100, help='num of datasets to train')

    args = parser.parse_args()
    # data_utils
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Result = {'Causal Forest':[]}

    for _ in range(num_dataset):

        cur_save_path = save_path + '/' + str(_)
        if not os.path.exists(cur_save_path):
            os.makedirs(cur_save_path)

        train_matrix = pd.read_csv(load_path + '/' + str(_) + '/train.txt', header=None, sep=' ').to_numpy()
        test_matrix = pd.read_csv(load_path + '/' + str(_) + '/test.txt', header=None, sep=' ').to_numpy()
        t_grid = pd.read_csv(load_path + '/' + str(_) + '/t_grid.txt', header=None, sep=' ').to_numpy()
        t = train_matrix[:, 0]
        x = train_matrix[:, 1:-3]
        y1 = train_matrix[:,-3]
        y2 = train_matrix[:,-1]
        idx = np.where(y1>0)

        model1 = CausalForest() #inference=False,fit_intercept=False
        model2 = CausalForest()
        model1.fit(x,t,y1)
        model2.fit(x[idx],t[idx],y2[idx])

        t_grid_hat, mse1, mse2= curve([model1,model2],test_matrix, t_grid)

        print(_,' current mse1: ', mse1,' mse2: ',mse2)
        Result['Causal Forest'].append([mse1,mse2])

        with open(save_path + '/result_2.json', 'w') as fp:
            json.dump(Result, fp)

