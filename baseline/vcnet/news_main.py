import torch
import math
import numpy as np
from models.dynamic_net import Vcnet, Drnet, TR, Vcnet_2
from data.data import get_iter
from utils.eval import curve, curve_2

import os
import json
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':
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

# ================= loss =================2
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * (torch.log(out[0] + epsilon) * y.squeeze()).mean()
    loss_y = (-y * torch.log(out[1] + epsilon).squeeze()
              - (1 - y) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
    return loss_pi + loss_y

def criterion_2(out, y1, y2, alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * torch.log(out[0] + epsilon).mean()
    loss_y = ((-y2 * torch.log(out[1] + epsilon).squeeze()
              - (1 - y2) * torch.log(1 - out[1] + epsilon).squeeze())
              * y1.squeeze()).mean()
    return loss_pi + loss_y

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze()
                    - trg.squeeze() / (out[0].squeeze() + epsilon)
                    - out[1].squeeze()) ** 2).mean()

def criterion_TR_2(out, trg, y1, y2, beta=1., epsilon=1e-6):
    return beta * (y1.squeeze()
                   * (y2.squeeze()
                      - trg.squeeze() / (out[0].squeeze() + epsilon)
                      - out[1].squeeze()) ** 2).mean()

# ================= main =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data_utils')

    parser.add_argument('--data_dir', type=str,
                        default='/root/test01/research/CausalCVR/dataset/news')
    parser.add_argument('--data_split_dir', type=str,
                        default='/root/test01/research/CausalCVR/dataset/news/eval')
    parser.add_argument('--save_dir', type=str,
                        default='logs/news/eval')
    parser.add_argument('--num_dataset', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=600)
    parser.add_argument('--verbose', type=int, default=100)

    args = parser.parse_args()

    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    tr_wd = 1e-5

    num_epoch = args.n_epochs
    verbose = args.verbose

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')

    
    Result = {}
    h = 32 

    for model_name in ['Vcnet_tr','Dragonnet_tr', 'Drnet', 'Tarnet']:
        Result[model_name] = []

        # ===== model init =====
        if model_name in ['Dragonnet_tr']:
            cfg_density = [(498, h, 1, 'relu'), (h, h, 1, 'relu')]
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            model1 = Drnet(cfg_density, 10, cfg, isenhance=0).to(device)
            model2 = Drnet(cfg_density, 10, cfg, isenhance=0).to(device)
            lr =5e-4
            lr_tr = 1e-3

        elif model_name in ['Drnet']:
            cfg_density = [(498, h, 1, 'relu'), (h, h, 1, 'relu')]
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            model1 = Drnet(cfg_density, 10, cfg, isenhance=1).to(device)
            model2 = Drnet(cfg_density, 10, cfg, isenhance=1).to(device)
            lr = 5e-4

        elif model_name in ['Tarnet']:
            cfg_density = [(498, h, 1, 'relu'), (h, h, 1, 'relu')]
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            model1 = Drnet(cfg_density, 10, cfg, isenhance=0).to(device)
            model2 = Drnet(cfg_density, 10, cfg, isenhance=0).to(device)
            lr = 5e-4

        else:
            cfg_density = [(498, h, 1, 'relu'), (h, h, 1, 'relu')]
            num_grid = 10
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            model1 = Vcnet(cfg_density, num_grid, cfg, 2, [0.33, 0.66]).to(device)
            model2 = Vcnet(cfg_density, num_grid, cfg, 2, [0.33, 0.66]).to(device)
            lr = lr_tr = 1e-5

        isTargetReg = model_name.endswith('_tr')

        if isTargetReg:
            TargetReg1 = TR(2, list(np.arange(0.1, 1, 0.1))).to(device)
            TargetReg2 = TR(2, list(np.arange(0.1, 1, 0.1))).to(device)

        alpha = 0.5 if isTargetReg else 0.0
        beta = 1.

        for d in range(args.num_dataset):
            print(f'----- Dataset {d}, Model {model_name} -----')
            idx_train = torch.load(f'{args.data_split_dir}/{d}/idx_train.pt')
            idx_test = torch.load(f'{args.data_split_dir}/{d}/idx_test.pt')

            train_matrix = data_matrix[idx_train].to(device)
            test_matrix = data_matrix[idx_test].to(device)
            t_grid = t_grid_all[:, idx_test].to(device)

            train_loader = get_iter(train_matrix, 500, True)

            model1._initialize_weights()
            model2._initialize_weights()

            optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr, #lr1,
                                            momentum=momentum, weight_decay=wd, nesterov=True)
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr,
                                            momentum=momentum, weight_decay=wd, nesterov=True)

            if isTargetReg:
                TargetReg1._initialize_weights()
                TargetReg2._initialize_weights()
                tr_optimizer1 = torch.optim.SGD(TargetReg1.parameters(), lr=5e-4,
                                                weight_decay=tr_wd)
                tr_optimizer2 = torch.optim.SGD(TargetReg2.parameters(), lr=lr_tr,
                                                weight_decay=tr_wd)

            for epoch in range(num_epoch):
                for inputs, y2 in train_loader:
                    t = inputs[:, 0]
                    x = inputs[:, 1:-2]
                    y1 = inputs[:, -2]

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    out1, out2 = model1(t, x), model2(t, x)

                    loss1 = criterion(out1, y1, alpha)
                    loss2 = criterion_2(out2, y1, y2, alpha)
                    # if torch.isnan(loss1) or torch.isnan(loss2):
                    #     print('nan detected.')
                    #     flag = 1

                    if isTargetReg:
                        trg1, trg2 = TargetReg1(t), TargetReg2(t)
                        loss1 += criterion_TR(out1, trg1, y1, beta)
                        loss2 += criterion_TR_2(out2, trg2, y1, y2, beta)

                    loss1.backward()
                    loss2.backward()
                    optimizer1.step()
                    optimizer2.step()


            if isTargetReg:
                _, mse1, mse2 = curve_2([model1, model2],
                                        test_matrix, t_grid,
                                        TargetReg1, TargetReg2)
            else:
                _, mse1, mse2 = curve_2([model1, model2],
                                        test_matrix, t_grid)

            mse1,mse2 = float(mse1),float(mse2)
            print('current loss: ', float(loss1.data),', ',float(loss2.data))
            print('current mse1: ', mse1,' mse2: ',mse2)

            Result[model_name].append([mse1,mse2])
            with open(args.save_dir + '/result_baseline.json', 'w') as f:
                json.dump(Result, f)

