import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd

import os
import json

from models.dynamic_net import CVRNet,Dynamic_FC
from data.data import get_iter
from utils.eval import curve_dr

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, model_name='', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def criterion_cvr(out, y1, y2,alpha=0.5, epsilon=1e-9):
    loss_pi = -alpha * torch.log(out[0] + epsilon).mean()
    out2 = out[1]*out[2]
    loss_y1 = (-y1 * torch.log(out[1] + epsilon).squeeze() - (1-y1) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
    loss_y2 = (-y2 * torch.log(out2 + epsilon).squeeze() - (1-y2) * torch.log(1 - out2 + epsilon).squeeze()).mean()
    return loss_pi + loss_y1 + loss_y2

def get_pseudo_out(y1, y2, out):
    pi,mu1,mu2 = out[0].detach(), out[1].squeeze().detach(), out[1].squeeze()*out[2].squeeze().detach()
    psi_1 = mu1 + ( (y1 - mu1) / (pi + 1e-6) ).detach()
    psi_2 = out[2].squeeze().detach() + ((y2-mu2)*mu1 - (y1-mu1)*mu2) / (pi * mu1**2 + 1e-6).detach()
    return torch.clamp(psi_1, 0.0, 1.0), torch.clamp(psi_2, 0.0, 1.0)


def train_nuisance(model, data_loader, n_epochs=500, lr=1e-5, verbose=100):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for ep in range(n_epochs):
        for batch, y2 in data_loader:
            t = batch[:,0]
            x = batch[:,1:-2]
            y1 = batch[:,-2]

            out = model(t,x)
            loss = criterion_cvr(out,y1,y2)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep%verbose==0:
            print(f"[Nuisance] epoch:{ep}, loss:{loss.item():.4f}")
    return model

def train_final(model, t, x, y, epochs=600, lr=1e-3):
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()

    for ep in range(epochs):
        pred = model(t,x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 100 == 0:
            print(f"[Final] epoch:{ep}, loss={loss.item():.10f}")
    return model


class FinalDynamicNet(nn.Module):
    def __init__(self, cfg, degree, knots):
        super().__init__()

        blocks = []
        for i, layer_cfg in enumerate(cfg):
            ind, outd, isbias, act = layer_cfg
            if i == len(cfg)-1:
                blocks.append(Dynamic_FC(ind, outd, degree, knots,
                                         act=act, isbias=isbias, islastlayer=1))
            else:
                blocks.append(Dynamic_FC(ind, outd, degree, knots,
                                         act=act, isbias=isbias, islastlayer=0))

        self.Q = nn.Sequential(*blocks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, x):
        tx = torch.cat([t.unsqueeze(1), x], dim=1)
        return self.sigmoid(self.Q(tx))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/simu2/eval', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='/root/test01/research/CausalCVR/logs/simu4/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=100, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=600, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    args = parser.parse_args()

    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # data_utils
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    Result = {}
    h = 8 
    init_lr = 5e-4 
    lr_final = 5e-3

    for model_name in ['DR']:
        cfg_density = [(8, h, 1, 'relu'), (h, h, 1, 'relu')]
        num_grid = 10
        cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
        degree = 2
        knots = [0.33, 0.66]
        Result[model_name] = []

        for _ in range(num_dataset):

            cur_save_path = save_path + '/' + str(_)
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            data = pd.read_csv(load_path + '/' + str(_) + '/train.txt', header=None, sep=' ')
            train_matrix = torch.from_numpy(data.to_numpy()).float().to(device)
            data = pd.read_csv(load_path + '/' + str(_) + '/test.txt', header=None, sep=' ')
            test_matrix = torch.from_numpy(data.to_numpy()).float().to(device)
            data = pd.read_csv(load_path + '/' + str(_) + '/t_grid.txt', header=None, sep=' ')
            t_grid = torch.from_numpy(data.to_numpy()).float().to(device)

            idx = torch.randperm(train_matrix.shape[0])
            n = train_matrix.shape[0]//2
            data_A = train_matrix[idx[:n]]
            data_B = train_matrix[idx[n:]]

            loader_A = get_iter(data_A,batch_size=500,shuffle=True)
            loader_B = get_iter(data_B,batch_size=500,shuffle=True)
            model1 = CVRNet(cfg_density,num_grid,cfg,degree,knots).to(device)
            model2 = CVRNet(cfg_density,num_grid,cfg,degree,knots).to(device)
            model1._initialize_weights()
            model2._initialize_weights()

            print(f"\n===== Training Nuisance Model1 on A =====")
            model1 = train_nuisance(model1,loader_A,num_epoch,init_lr,verbose)
            print(f"===== Training Nuisance Model2 on B =====")
            model2 = train_nuisance(model2,loader_B,num_epoch,init_lr,verbose)

            with torch.no_grad():
                tA, xA, y1A, y2A = data_A[:,0], data_A[:,1:-3], data_A[:,-3], data_A[:,-1]
                tB, xB, y1B, y2B = data_B[:,0], data_B[:,1:-3], data_B[:,-3], data_B[:,-1]
                out1_on_B = model1(tB,xB)     # model1 → B
                out2_on_A = model2(tA,xA)     # model2 → A
                psi1_A,psi2_A = get_pseudo_out(y1A,y2A,out2_on_A)
                psi1_B,psi2_B = get_pseudo_out(y1B,y2B,out1_on_B)
                t_all = torch.cat([tA,tB],dim=0)
                X_all  = torch.cat([xA,xB],dim=0)
                PSI1   = torch.cat([psi1_A,psi1_B],dim=0).unsqueeze(1)
                PSI2   = torch.cat([psi2_A,psi2_B],dim=0).unsqueeze(1)
                print('***',PSI1.mean(), PSI2.mean())

            final_cfg = [(8, h, 1, 'relu'),(h, h, 1, 'relu'),(h, 1, 1, 'id')]
            final_model1 = FinalDynamicNet(final_cfg, degree, knots).to(device)
            final_model2 = FinalDynamicNet(final_cfg, degree, knots).to(device)
            final_model1._initialize_weights()
            final_model2._initialize_weights()
            print(f"\n===== Training Final Model1 =====")
            final_model1 = train_final(final_model1,t_all,X_all,PSI1)
            print(f"\n===== Training Final Model2 =====")
            final_model2 = train_final(final_model2,t_all,X_all,PSI2)

            t_grid_hat, mse1, mse2 = curve_dr([final_model1,final_model2],test_matrix, t_grid)
            mse1,mse2 = float(mse1),float(mse2)
            print('current mse1: ', mse1,' mse2: ',mse2)
            Result[model_name].append([mse1,mse2])
            with open(save_path + '/result_dr.json', 'w') as fp:
                json.dump(Result, fp)
                    