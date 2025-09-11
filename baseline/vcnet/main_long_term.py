import sys
sys.path.append('/home/luban/yangzeqin/Project')

import random
import torch
import math
import numpy as np
from models.dynamic_net import Vcnet, Drnet, TR
from torch.utils.data import DataLoader, TensorDataset
from utils.eval import curve, test

from data.data_utils import read_data_withU

import os
import argparse

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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

def save_checkpoint(state, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/home/luban/yangzeqin/Project/data/simulation3/', help='dir of data_utils matrix')
    parser.add_argument('--save_dir', type=str, default='/home/luban/yangzeqin/Project/baseline/vcnet/logs/', help='dir to save result')
    parser.add_argument('--betaU', type=float, default=0.25, help='data_utils file')

    # training
    parser.add_argument('--model', type=str, default='Vcnet_tr', help='model')  # Vcnet, Vcnet_tr
    parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs to train')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=int, default=3, help='gpu number')
    parser.add_argument('--batch_size', type=int, default=512, help='gpu number')
    parser.add_argument('--init_lr', type=float, default=5e-3, help='learning rate')


    # print train info
    parser.add_argument('--verbose', type=int, default=5, help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=False, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.cuda.set_device(args.device)
    set_seed(args.seed)

    # Parameters

    # optimizer
    lr_type = 'fixed'
    wd = 5e-4
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-5

    # epoch: 800!
    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # get data_utils
    betaU = args.betaU
    log_file = open("logs/result_betaU{}.txt".format(betaU), "a", encoding="utf-8")

    test_loss_y_list, test_loss_cfy_list = [], []
    for file_seed in range(0, 10):
        print("========== begin seed={} ==========".format(file_seed))

        file_name = args.data_dir + "data_p15_q5_7to14_betaU{}_seed{}.npz".format(betaU, file_seed)

        print("prepare dataset...")
        exp_data, train_obs_data, valid_obs_data, test_obs_data = read_data_withU(file_name)
        X_e, T_e, S_e, Y_e, G_e, U_e = exp_data
        train_X_o, train_T_o, train_S_o, train_Y_o, train_G_o, train_U_o = train_obs_data
        valid_X_o, valid_T_o, valid_S_o, valid_Y_o, valid_cfT_o, valid_cfY_o = valid_obs_data
        test_X_o, test_T_o, test_S_o, test_Y_o, test_cfT_o, test_cfY_o = test_obs_data

        X_e, T_e, S_e, Y_e, G_e = torch.tensor(X_e).cuda(), torch.tensor(T_e).cuda(), torch.tensor(S_e).cuda(), torch.tensor(Y_e).cuda(), torch.tensor(G_e).cuda()
        train_X_o, train_T_o, train_S_o, train_Y_o, train_G_o, train_U_o = torch.tensor(train_X_o).cuda(), torch.tensor(train_T_o).cuda(), torch.tensor(train_S_o).cuda(), torch.tensor(train_Y_o).cuda(), torch.tensor(train_G_o).cuda(), torch.tensor(train_U_o).cuda()
        valid_X_o, valid_T_o, valid_S_o, valid_Y_o, valid_cfT_o, valid_cfY_o = torch.tensor(valid_X_o).cuda(), torch.tensor(valid_T_o).cuda(), torch.tensor(valid_S_o).cuda(), torch.tensor(valid_Y_o).cuda(), torch.tensor(valid_cfT_o).cuda(), torch.tensor(valid_cfY_o).cuda()
        test_X_o, test_T_o, test_S_o, test_Y_o, test_cfT_o, test_cfY_o = (torch.tensor(test_X_o).cuda(), torch.tensor(test_T_o).cuda(), torch.tensor(test_S_o).cuda(),torch.tensor(test_Y_o).cuda(), torch.tensor(test_cfT_o).cuda(), torch.tensor(test_cfY_o).cuda())

        train_o_dataset = TensorDataset(train_X_o, train_T_o, train_Y_o)

        train_loader = DataLoader(train_o_dataset, batch_size=args.batch_size, shuffle=True)

        save_path = ''
        # for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr']:
        for model_name in [args.model]:
            # import model
            if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
                cfg_density = [(15, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                degree = 2
                knots = [0.33, 0.66]
                model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
                model._initialize_weights()

            elif model_name == 'Drnet' or model_name == 'Drnet_tr':
                cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                isenhance = 1
                model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
                model._initialize_weights()

            elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
                cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                isenhance = 0
                model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
                model._initialize_weights()

            # use Target Regularization?
            if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
                isTargetReg = 1
            else:
                isTargetReg = 0

            #if isTargetReg:
            tr_knots = list(np.arange(0.05, 1, 0.05))
            tr_degree = 2
            TargetReg = TR(tr_degree, tr_knots)
            TargetReg._initialize_weights()

            # best cfg for each model
            if model_name == 'Tarnet':
                init_lr = 0.02
                alpha = 1.0
                tr_init_lr = 0.001
                beta = 1.
            elif model_name == 'Tarnet_tr':
                init_lr = 0.02
                alpha = 0.5
                tr_init_lr = 0.001
                beta = 1.
            elif model_name == 'Drnet':
                init_lr = 0.02
                alpha = 1.
                tr_init_lr = 0.001
                beta = 1.
            elif model_name == 'Drnet_tr':
                init_lr = 0.02
                alpha = 0.5
                tr_init_lr = 0.001
                beta = 1.
            elif model_name == 'Vcnet':
                init_lr = args.init_lr
                alpha = 0.5
                tr_init_lr = args.init_lr
                beta = 1.
            elif model_name == 'Vcnet_tr':
                # init_lr = 0.0001
                init_lr = args.init_lr
                alpha = 0.1
                tr_init_lr = args.init_lr
                beta = 1.

            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)
            # optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=wd)

            if isTargetReg:
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
                # tr_optimizer = torch.optim.Adam(params=TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            model.cuda()
            TargetReg.cuda()

            print('model = ', model_name)
            min_test_f_loss = float('inf')
            min_test_cf_loss = float('inf')
            for epoch in range(num_epoch):

                for idx, (x, t, y) in enumerate(train_loader):

                    if isTargetReg:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        loss_y, loss_tar = criterion(out, y, alpha=alpha), criterion_TR(out, trg, y, beta=beta)
                        loss = loss_y + loss_tar
                        loss.backward()
                        optimizer.step()

                        tr_optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        tr_loss = criterion_TR(out, trg, y, beta=beta)
                        tr_loss.backward()
                        tr_optimizer.step()
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion(out, y, alpha=alpha)
                        loss.backward()
                        optimizer.step()

                if isTargetReg:
                    loss_y, loss_cfy = test(model, test_X_o, test_T_o, test_Y_o, test_cfT_o, test_cfY_o, TargetReg)
                else:
                    loss_y, loss_cfy = test(model, test_X_o, test_T_o, test_Y_o, test_cfT_o, test_cfY_o)
                if loss_cfy <= min_test_cf_loss:
                    min_test_f_loss = loss_y
                    min_test_cf_loss = loss_cfy

                if epoch % verbose == 0:
                    print("epoch:{}, loss_test_y: {:.4f}, loss_test_cfy: {:.4f}".format(epoch, loss_y, loss_cfy))

            if isTargetReg:
                loss_y, loss_cfy = test(model, test_X_o, test_T_o, test_Y_o, test_cfT_o, test_cfY_o, TargetReg)
            else:
                loss_y, loss_cfy = test(model, test_X_o, test_T_o, test_Y_o, test_cfT_o, test_cfY_o)

            test_loss_y_list.append(loss_y)
            test_loss_cfy_list.append(loss_cfy)

            print("seed: {}, loss_test_y: {:.4f}, loss_test_cfy: {:.4f}".format(file_seed, loss_y, loss_cfy))

    print("loss_list: {}".format([round(x, 5) for x in test_loss_cfy_list]))
    print("f_loss_mean: {:.4f}, f_loss_std: {:.4f}".format(np.array(test_loss_y_list).mean(), np.array(test_loss_y_list).std()))
    print("cf_loss_mean: {:.4f}, cf_loss_std: {:.4f}".format(np.array(test_loss_cfy_list).mean(), np.array(test_loss_cfy_list).std()))
    log_file.close()
