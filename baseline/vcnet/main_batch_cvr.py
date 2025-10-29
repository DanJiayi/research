import torch
import math
import numpy as np
import pandas as pd

import os
import json

from models.dynamic_net import Vcnet, Drnet, TR, Vcnet_2
from data.data import get_iter
from utils.eval import curve,curve_2

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

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * torch.log(out[0] + epsilon).mean()
    loss_y = (-y * torch.log(out[1] + epsilon).squeeze() - (1-y) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
    return loss_pi + loss_y

def criterion_2(out, y1, y2,alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * (torch.log(out[0] + epsilon)* y1.squeeze()).mean()
    # loss_y1 = (-y1 * torch.log(out[1] + epsilon).squeeze() - (1-y1) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
    loss_y = ((-y2 * torch.log(out[1] + epsilon).squeeze() - (1-y2) * torch.log(1 - out[1] + epsilon).squeeze()) * y1.squeeze()).mean()
    return loss_pi + loss_y 

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

def criterion_TR_2(out, trg, y1, y2,beta=1., epsilon=1e-6):
    #return beta *  (y1.squeeze() *(y2.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[2].squeeze())**2).mean()
    return beta *  (y1.squeeze() *(y2.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/simu2/eval', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu4/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=100, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=600, help='num of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-7, help='lr')
    parser.add_argument('--lr_tr', type=float, default=1e-6, help='lr_tr')

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
    for model_name in ['Drnet']: #'Vcnet_tr','Vcnet',,'Tarnet','Vcnet_tr','Dragonnet_tr',
    #for model_name in ['Vcnet_tr']:
        h = 32
        lr1 = 1e-5
        lr = args.lr
        lr_tr = args.lr_tr
        Result[model_name]=[]
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(8, h, 1, 'relu'), (h, h, 1, 'relu')]
            num_grid = 10
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            #model = Vcnet_2(cfg_density, num_grid, cfg, degree, knots).to(device)
            model1 = Vcnet(cfg_density, num_grid, cfg, degree, knots).to(device)
            model2 = Vcnet(cfg_density, num_grid, cfg, degree, knots).to(device)
            model1._initialize_weights()
            model2._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(8, h, 1, 'relu'), (h, h, 1, 'relu')]
            num_grid = 10
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            isenhance = 1
            model1 = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance).to(device)
            model1._initialize_weights()
            model2 = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance).to(device)
            model2._initialize_weights()

        elif model_name == 'Dragonnet' or model_name == 'Dragonnet_tr' or model_name == 'Tarnet':
            cfg_density = [(8, h, 1, 'relu'), (h, h, 1, 'relu')]
            num_grid = 10
            cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
            isenhance = 0
            model1 = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance).to(device)
            model1._initialize_weights()
            model2 = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance).to(device)
            model2._initialize_weights()

        # use Target Regularization
        if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Dragonnet_tr':
            isTargetReg = 1
        else:
            isTargetReg = 0

        if isTargetReg:
            tr_knots = list(np.arange(0.1, 1, 0.1))
            tr_degree = 2
            TargetReg1 = TR(tr_degree, tr_knots).to(device)
            TargetReg2 = TR(tr_degree, tr_knots).to(device)
            TargetReg1._initialize_weights()
            TargetReg2._initialize_weights()

        # best cfg for each model
        if model_name == 'Dragonnet':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.5

            Result['Dragonnet'] = []

        elif model_name == 'Dragonnet_tr':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.5
            tr_init_lr = 1e-6
            beta = 1.

            Result['Dragonnet_tr'] = []

        elif model_name == 'Tarnet':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.0

            Result['Tarnet'] = []

        elif model_name == 'Drnet':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.0 #0.5

            Result['Drnet'] = []

        elif model_name == 'Drnet_tr':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.5
            tr_init_lr = 1e-6
            beta = 1.

            Result['Drnet_tr'] = []

        elif model_name == 'Vcnet':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.5

            Result['Vcnet'] = []

        elif model_name == 'Vcnet_tr':
            init_lr1 = lr1
            init_lr2 = 1e-6
            alpha = 0.5
            tr_init_lr = 1e-6
            beta = 1.

            Result['Vcnet_tr'] = []

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

            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
            train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # reinitialize model
            model1._initialize_weights()
            model2._initialize_weights()

            # define optimizer
            optimizer1 = torch.optim.SGD(model1.parameters(), lr=init_lr1, momentum=momentum, weight_decay=wd, nesterov=True)
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=init_lr2, momentum=momentum, weight_decay=wd, nesterov=True)

            if isTargetReg:
                TargetReg1._initialize_weights()
                TargetReg2._initialize_weights()
                tr_optimizer1 = torch.optim.SGD(TargetReg1.parameters(), lr=1e-3, weight_decay=tr_wd)
                tr_optimizer2 = torch.optim.SGD(TargetReg2.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            print('model : ', model_name)
            for epoch in range(num_epoch):
                for idx, (inputs,y2) in enumerate(train_loader):
                    t = inputs[:, 0]
                    x = inputs[:, 1:-2]
                    y1 = inputs[:,-2]

                    if isTargetReg:
                        optimizer1.zero_grad()
                        out1,out2 = model1.forward(t, x),model2.forward(t, x)
                        trg1,trg2 = TargetReg1(t),TargetReg2(t)
                        loss1 = criterion(out1, y1,alpha=alpha) + criterion_TR(out1, trg1, y1, beta=beta)
                        loss1.backward()
                        optimizer1.step()
                        loss2 = criterion_2(out2, y1,y2,alpha=alpha) + criterion_TR_2(out2, trg2, y1, y2, beta=beta)
                        loss2.backward()
                        optimizer2.step()

                        tr_optimizer1.zero_grad()
                        tr_optimizer2.zero_grad()
                        out1,out2 = model1.forward(t, x),model2.forward(t, x)
                        trg1, trg2 = TargetReg1(t), TargetReg2(t)
                        tr_loss1 = criterion_TR(out1, trg1, y1, beta=beta)
                        tr_loss2 = criterion_TR_2(out2, trg2, y1, y2, beta=beta)
                        tr_loss1.backward(retain_graph=True) 
                        tr_optimizer1.step()
                        tr_loss2.backward() 
                        tr_optimizer2.step()
                    else:
                        optimizer1.zero_grad()
                        out1,out2 = model1.forward(t, x),model2.forward(t, x)                     
                        loss1 = criterion(out1, y1,alpha=alpha)
                        loss1.backward()
                        optimizer1.step()
                        loss2 = criterion_2(out2, y1,y2,alpha=alpha)
                        loss2.backward()
                        optimizer2.step()

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss1.data,', ',loss2.data)

            if isTargetReg:
                t_grid_hat, mse1, mse2 = curve_2([model1,model2],test_matrix, t_grid, TargetReg1,TargetReg2)
            else:
                t_grid_hat, mse1, mse2 = curve_2([model1,model2], test_matrix, t_grid)

            mse1,mse2 = float(mse1),float(mse2)
            print('current loss: ', float(loss1.data),', ',float(loss2.data))
            print('current mse1: ', mse1,' mse2: ',mse2)
            # print('-----------------------------------------------------------------')
            # save_checkpoint({
            #     'model': model_name,
            #     'best_test_loss': [mse1,mse2],
            #     'model_state_dict': [model1.state_dict(),model2.state_dict()],
            #     'TR_state_dict': [TargetReg1.state_dict(),TargetReg2.state_dict()] if isTargetReg else None
            # }, model_name=model_name, checkpoint_dir=cur_save_path)
            # print('-----------------------------------------------------------------')

            Result[model_name].append([mse1,mse2])

            with open(save_path + f'/result_baseline_corrected_2.json', 'w') as fp:
                json.dump(Result, fp)