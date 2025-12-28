import torch
import math
import numpy as np
import pandas as pd
from models.dynamic_net import Dragonnet_Binary,TR_Binary,T_Learner
from data.data import get_iter
from utils.eval import eval_binary

import os
import json
import argparse
import sys

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

def save_checkpoint(state, model_name, checkpoint_dir):
    if checkpoint_dir.endswith('.pth.tar'):
        filename = checkpoint_dir
    else:
        filename = os.path.join(checkpoint_dir, f"{model_name}_ckpt.pth.tar")

    torch.save(state, filename)
    print(f"=> Saving checkpoint to {filename}")

# criterion
def criterion(out, t,y, alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * ((-t * torch.log(out[0] + epsilon).squeeze() - (1-t) * torch.log(1 - out[0] + epsilon).squeeze()).mean())
    mu = (1-t)*out[1][0] + t*out[1][1]
    loss_y = (-y * torch.log(mu + epsilon).squeeze() - (1-y) * torch.log(1 - mu + epsilon).squeeze()).mean()
    return loss_pi + loss_y

def criterion_2(out, t,y1, y2,alpha=0.5, epsilon=1e-6):
    loss_pi = -alpha * (((-t * torch.log(out[0] + epsilon).squeeze() - (1-t) * torch.log(1 - out[0] + epsilon).squeeze())* y1.squeeze()).mean())
    mu = (1-t)*out[1][0] + t*out[1][1]
    loss_y = ((-y2 * torch.log(mu + epsilon).squeeze() - (1-y2) * torch.log(1 - mu + epsilon).squeeze()) * y1.squeeze()).mean()
    return loss_pi + loss_y 

def criterion_TR(out, t,trg, y, beta=1., epsilon=1e-9,detach=False):
    pi = out[0].detach() if detach else out[0]
    mu = (1-t)*out[1][0] + t*out[1][1] + trg * (t/(pi+epsilon) - (1-t)/(1-pi+epsilon))
    return beta *((mu.squeeze() - y.squeeze())**2).mean()

def criterion_TR_2(out, t,trg, y1, y2,beta=1., epsilon=1e-6):
    pi = out[0]
    mu = (1-t)*out[1][0] + t*out[1][1] + trg * (t/(pi+epsilon) - (1-t)/(1-pi+epsilon))
    return beta *(y1*(mu.squeeze() - y2.squeeze())**2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/criteo', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='/root/test01/research/baseline/vcnet/logs/criteo', help='dir to save result')

    # common
    # parser.add_argument('--num_dataset', type=int, default=100, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=40, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=1, help='print train info freq')

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
    # num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    log_path = os.path.join(save_path, "log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    class Logger(object):
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log = open(log_file, "a", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            pass
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout


    Result = {}
    for model_name in ['Dragonnet_tr','Tarnet','TLearner']: #,'Tarnet','TLearner'
        Result[model_name] = {}
        best_auuc,best_qini = 0,0
        best_params = None
        for h in [8,16,32,64,128,256]:
            for init_lr in [1e-6,1e-5,5e-5,1e-4,5e-4,1e-3]: #1e-6,
                for tr_init_lr in [1e-6,1e-5,5e-5,1e-4,5e-4,1e-3]:
                    if init_lr not in [1e-6,5e-5,5e-4] and tr_init_lr not in [1e-6,5e-5,5e-4] and h not in [128,256]: continue
                    flag = 0
                    print(f'{h}_{init_lr}_{tr_init_lr}')
                    if model_name == 'Dragonnet_tr' or  model_name == 'Tarnet':
                        cfg_density = [(12, h, 1, 'relu'), (h, h, 1, 'relu')]
                        cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
                        model1 = Dragonnet_Binary(cfg_density, cfg).to(device)
                        model1._initialize_weights()
                        model2 = Dragonnet_Binary(cfg_density, cfg).to(device)
                        model2._initialize_weights()

                    elif model_name == 'TLearner':
                        cfg = [(12, h, 1, 'relu'), (h, h, 1, 'relu'),(h, h, 1, 'relu'),(h, 1, 1, 'id')]
                        model1 = T_Learner(cfg).to(device)
                        model1._initialize_weights()
                        model2 = T_Learner(cfg).to(device)
                        model2._initialize_weights()


                    # use Target Regularization
                    if model_name == 'Dragonnet_tr':
                        isTargetReg = 1
                    else:
                        isTargetReg = 0

                    if isTargetReg:
                        TargetReg1 = TR_Binary().to(device)
                        TargetReg2 = TR_Binary().to(device)
                        TargetReg1._initialize_weights()
                        TargetReg2._initialize_weights()

                    # best cfg for each model
                    if model_name == 'Dragonnet_tr':
                        init_lr1 = init_lr
                        init_lr2 = init_lr
                        alpha = 0.5
                        tr_init_lr = tr_init_lr #1e-6
                        beta = 1.
                        # Result['Dragonnet_tr'] = []

                    elif model_name == 'Tarnet':
                        init_lr1 = init_lr
                        init_lr2 = init_lr
                        alpha = 0.0
                        # Result['Tarnet'] = []

                    elif model_name == 'TLearner':
                        init_lr1 = init_lr
                        init_lr2 = init_lr
                        alpha = 0.0
                        # Result['TLearner'] = []


                    cur_save_path = save_path #+ '/' + str(_)
                    if not os.path.exists(cur_save_path):
                        os.makedirs(cur_save_path)

                    data = pd.read_csv(load_path + '/train.csv')
                    train_matrix = torch.from_numpy(data.to_numpy()).float().to(device)
                    data = pd.read_csv(load_path + '/test.csv')
                    test_matrix = torch.from_numpy(data.to_numpy()).float().to(device)
                        
                    # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
                    train_loader = get_iter(train_matrix, batch_size=1024, shuffle=True)
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
                        if flag==1: continue
                        for idx, (inputs,exposure) in enumerate(train_loader):
                            t = inputs[:, 12]
                            x = inputs[:, :12]
                            y1 = inputs[:, 14]
                            y2 = inputs[:, 13]

                            if isTargetReg:
                                optimizer1.zero_grad()
                                out1,out2 = model1.forward(x),model2.forward(x)
                                trg1,trg2 = TargetReg1(x),TargetReg2(x)
                                loss1 = criterion(out1, t,y1,alpha=alpha) + criterion_TR(out1, t,trg1, y1, beta=beta)
                                loss1.backward()
                                optimizer1.step()
                                loss2 = criterion_2(out2, t,y1,y2,alpha=alpha) + criterion_TR_2(out2, t,trg2, y1, y2, beta=beta)
                                loss2.backward()
                                optimizer2.step()

                                tr_optimizer1.zero_grad()
                                tr_optimizer2.zero_grad()
                                out1,out2 = model1.forward(x),model2.forward(x)
                                trg1, trg2 = TargetReg1(x), TargetReg2(x)
                                tr_loss1 = criterion_TR(out1, t,trg1, y1, beta=beta)
                                tr_loss2 = criterion_TR_2(out2, t,trg2, y1, y2, beta=beta)
                                tr_loss1.backward(retain_graph=True) 
                                tr_optimizer1.step()
                                tr_loss2.backward() 
                                tr_optimizer2.step()
                            else:
                                optimizer1.zero_grad()
                                out1,out2 = model1.forward(x),model2.forward(x)                     
                                loss1 = criterion(out1, t,y1,alpha=alpha)
                                loss1.backward()
                                optimizer1.step()
                                loss2 = criterion_2(out2, t,y1,y2,alpha=alpha)
                                loss2.backward()
                                optimizer2.step()

                        epoch += 1
                        print('current epoch: ', epoch)
                        print('current loss: ', [loss1.data,loss2.data])
                        if torch.isnan(loss1) or torch.isnan(loss2): flag = 1
                        if epoch % 5 == 0:
                            if isTargetReg:
                                auuc1, auuc2, qini1, qini2 = eval_binary([model1,model2],test_matrix, TargetReg1,TargetReg2)
                            else:
                                auuc1, auuc2, qini1, qini2 = eval_binary([model1,model2], test_matrix)

                            params = f'{epoch}_{h}_{init_lr}_{tr_init_lr}'
                            Result[model_name][params]=[auuc1,auuc2,qini1,qini2]
                            
                            if auuc2>0 and qini2>0 and (auuc2+qini2) > (best_auuc+best_qini):
                                best_auuc,best_qini = auuc2,qini2
                                best_params = params

                            print('current auuc: ', [auuc1,auuc2],' current qini: ',[qini1,qini2],' best auuc: ',best_auuc,'best_qini: ',best_qini,' best_params: ',best_params)


                            with open(save_path + f'/result_baseline_test1.json', 'w') as fp:
                                json.dump(Result, fp)
