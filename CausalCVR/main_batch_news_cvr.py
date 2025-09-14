import torch
import math
import numpy as np
from models.dynamic_net import Vcnet, Drnet, TR, Vcnet_2
from data.data import get_iter
from utils.eval import curve,curve_2

import os
import json
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
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_2(out, y1, y2,alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * torch.log(out[0] + epsilon).mean()
    loss_y1 = (-y1 * torch.log(out[1] + epsilon).squeeze() - (1-y1) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
    loss_y2 = ((-y2 * torch.log(out[2] + epsilon).squeeze() - (1-y2) * torch.log(1 - out[2] + epsilon).squeeze()) * y1.squeeze()).mean()
    return loss_pi + loss_y1 + loss_y2 

def criterion_cvr(out, y1, y2,alpha=0.5, epsilon=1e-9):
    loss_pi = -alpha * torch.log(out[0] + epsilon).mean()
    out2 = out[1]*out[2]
    loss_y1 = (-y1 * torch.log(out[1] + epsilon).squeeze() - (1-y1) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
    loss_y2 = (-y2 * torch.log(out2 + epsilon).squeeze() - (1-y2) * torch.log(1 - out2 + epsilon).squeeze()).mean()
    return loss_pi + loss_y1 + loss_y2    

def criterion_TR(out, trg, y, beta=1., epsilon=1e-9):
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

def criterion_TR_2(out, trg, y1, y2,beta=1., epsilon=1e-9):
    return beta *  (y1.squeeze() *(y2.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[2].squeeze())**2).mean()

def criterion_TR_cvr(out, trg, y1, y2, beta=1., epsilon=1e-9):
    out1,out2 = out[1],out[1]*out[2]
    y1,y2,out1,out2,trg = y1.squeeze(),y2.squeeze(),out1.squeeze(),out2.squeeze(),trg.squeeze()
    return beta * (( (y2 - out2)/(out1 + epsilon) - (y1-out1)*out2/(out1**2+epsilon) - trg/(out[0].squeeze()+epsilon) )**2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/news', help='dir of data_utils matrix')
    parser.add_argument('--data_split_dir', type=str, default='/root/test01/research/CausalCVR/dataset/news/eval', help='dir of data_utils split')
    parser.add_argument('--save_dir', type=str, default='logs/news/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=10, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    args = parser.parse_args()

    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Parameters

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-5

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # load
    num_dataset = args.num_dataset
    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')

    Result = {}

    #for model_name in ['Vcnet', 'Vcnet_tr', 'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr']:
    for model_name in ['Vcnet', 'Vcnet_tr']:
        Result[model_name]=[]
        # import model
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet_2(cfg_density, num_grid, cfg, degree, knots).to(device)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance).to(device)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance).to(device)
            model._initialize_weights()

        # use Target Regularization?
        if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
            isTargetReg = 1
        else:
            isTargetReg = 0

        tr_knots=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        tr_degree = 2
        TargetReg1 = TR(tr_degree, tr_knots).to(device)
        TargetReg1._initialize_weights()
        TargetReg2 = TR(tr_degree, tr_knots).to(device)
        TargetReg2._initialize_weights()

        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.0005
            alpha = 1.0
            tr_init_lr = 0.001
            beta = 1.

            Result['Tarnet'] = []

        elif model_name == 'Tarnet_tr':
            init_lr = 0.0005
            alpha = 0.5
            tr_init_lr = 0.0005
            beta = 1.

            Result['Tarnet_tr'] = []

        elif model_name == 'Drnet':
            init_lr = 0.0005  # 0.005
            alpha = 1.
            tr_init_lr = 0.0005
            beta = 1.

            Result['Drnet'] = []

        elif model_name == 'Drnet_tr':
            init_lr = 0.0005 # 0.005
            alpha = 0.5
            tr_init_lr = 0.0005
            beta = 1.

            Result['Drnet_tr'] = []

        elif model_name == 'Vcnet':
            init_lr = 0.0005
            alpha = 0.5
            tr_init_lr = 0.0005
            beta = 1.

            Result['Vcnet'] = []

        elif model_name == 'Vcnet_tr':
            init_lr = 0.0005
            alpha = 0.5
            tr_init_lr = 0.0005
            beta = 0.5

            Result['Vcnet_tr'] = []

        for _ in range(num_dataset):
            cur_save_path = save_path + '/' + str(_)
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            idx_train = torch.load(args.data_split_dir + '/' + str(_) + '/idx_train.pt')
            idx_test = torch.load(args.data_split_dir + '/' + str(_) + '/idx_test.pt')

            train_matrix = data_matrix[idx_train, :].to(device)
            test_matrix = data_matrix[idx_test, :].to(device)
            t_grid = t_grid_all[:, idx_test].to(device)

            train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # reinitialize model
            model._initialize_weights()

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd,
                                        nesterov=True)

            if isTargetReg:
                TargetReg1._initialize_weights()
                TargetReg2._initialize_weights()
                tr_optimizer1 = torch.optim.SGD(TargetReg1.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
                tr_optimizer2 = torch.optim.SGD(TargetReg2.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            print('model : ', model_name)
            for epoch in range(num_epoch):
                for idx, (inputs, y2) in enumerate(train_loader):
                    t = inputs[:, 0]
                    x = inputs[:, 1:-2]
                    y1 = inputs[:,-2]

                    if isTargetReg:
                        if epoch <= 800:
                            optimizer.zero_grad()
                            out = model.forward(t, x)
                            trg1,trg2 = TargetReg1(t),TargetReg2(t)
                            loss = criterion_cvr(out, y1,y2,alpha=alpha) + criterion_TR(out, trg1, y1, beta=beta) + criterion_TR_cvr(out, trg2, y1, y2, beta=beta)
                            loss.backward()
                            optimizer.step()

                        tr_optimizer1.zero_grad()
                        tr_optimizer2.zero_grad()
                        out = model(t, x)
                        trg1, trg2 = TargetReg1(t), TargetReg2(t)
                        tr_loss1 = criterion_TR(out, trg1, y1, beta=beta)
                        tr_loss2 = criterion_TR_cvr(out, trg2, y1, y2, beta=beta)
                        tr_loss1.backward(retain_graph=True) 
                        tr_optimizer1.step()
                        tr_loss2.backward() 
                        tr_optimizer2.step()
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion_cvr(out, y1,y2,alpha=alpha)
                        loss.backward()
                        optimizer.step()

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss)

            if isTargetReg:
                t_grid_hat, mse1, mse2 = curve_2(model, test_matrix, t_grid, TargetReg1, TargetReg2)
                mse1, mse2 = float(mse1),float(mse2)
                print('current loss: ', float(loss.data))
                print('current mse1: ', mse1,' mse2: ',mse2)
            else:
                t_grid_hat, mse1, mse2 = curve_2(model, test_matrix, t_grid, TargetReg1, TargetReg2)
                mse1, mse2 = float(mse1),float(mse2)
                print('current loss: ', float(loss.data))
                print('current mse1: ', mse1,' mse2: ',mse2)

            print('-----------------------------------------------------------------')
            save_checkpoint({
                'model': model_name,
                'best_test_loss': [mse1,mse2],
                'model_state_dict': model.state_dict(),
                'TR_state_dict': [TargetReg1.state_dict(),TargetReg2.state_dict()],
            }, model_name=model_name, checkpoint_dir=cur_save_path)
            print('-----------------------------------------------------------------')

            Result[model_name].append([mse1,mse2])

            with open(save_path + '/result.json', 'w') as fp:
                json.dump(Result, fp)


