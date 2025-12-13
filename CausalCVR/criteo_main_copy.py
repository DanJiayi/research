import torch
import math
import numpy as np
import pandas as pd

import os
import json

from models.dynamic_net import CVRNet_Binary,TR_Binary
from data.data import get_iter
from utils.eval import curve,curve_2,eval_binary_2
from sklift.metrics import uplift_auc_score, qini_auc_score

import argparse
import sys
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

set_seed(42)

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

# def save_checkpoint(state, model_name='', checkpoint_dir='.'):
#     filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
#     print('=> Saving checkpoint to {}'.format(filename))
#     torch.save(state, filename)

def save_checkpoint(state, model_name, checkpoint_dir):
    if checkpoint_dir.endswith('.pth.tar'):
        filename = checkpoint_dir
    else:
        filename = os.path.join(checkpoint_dir, f"{model_name}_ckpt.pth.tar")

    torch.save(state, filename)
    print(f"=> Saving checkpoint to {filename}")

# criterion
# def criterion(out, y, t,alpha=0.5, epsilon=1e-9):
#     p = out[0]
#     mu0,mu1 = out[1][0],out[1][1]
#     # mu20,mu21 = out[1][0]*out[2][0],out[1][1]*out[2][1]
#     loss_pi = 0
#     return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_2(out, t,y1, y2,alpha=0.5, epsilon=1e-9):
    mu1 = (1-t)*out[1][0] + t*out[1][0]
    mu2 = (1-t)*out[1][0]*out[2][0] + t*out[1][1]*out[2][1]
    loss_pi = -alpha * ((-t * torch.log(out[0] + epsilon).squeeze() - (1-t) * torch.log(1 - out[0] + epsilon).squeeze()).mean())
    loss_y1 = (-y1 * torch.log(mu1  + epsilon).squeeze() - (1-y1) * torch.log(1 - mu1  + epsilon).squeeze()).mean()
    loss_y2 = ((-y2 * torch.log(mu2 + epsilon).squeeze() - (1-y2) * torch.log(1 - mu2 + epsilon).squeeze()) * y1.squeeze()).mean()
    return loss_pi + 0*loss_y1 + loss_y2 

def criterion_cvr(out, t,y1, y2,alpha=0.5, epsilon=1e-9):
    mu1 = (1-t)*out[1][0] + t*out[1][1]
    mu2 = (1-t)*out[1][0]*out[2][0] + t*out[1][1]*out[2][1]
    loss_pi = -alpha * ((-t * torch.log(out[0] + epsilon).squeeze() - (1-t) * torch.log(1 - out[0] + epsilon).squeeze()).mean())
    loss_y1 = (-y1 * torch.log(mu1 + epsilon).squeeze() - (1-y1) * torch.log(1 - mu1 + epsilon).squeeze()).mean()
    loss_y2 = (-y2 * torch.log(mu2 + epsilon).squeeze() - (1-y2) * torch.log(1 - mu2 + epsilon).squeeze()).mean()
    return loss_pi + 0*loss_y1 + loss_y2

# def criterion_id(out, y1, y2,alpha=0.5, epsilon=1e-9):
#     loss_pi = -alpha * torch.log(out[0] + epsilon).mean()
#     loss_y1 = (-y1 * torch.log(out[1] + epsilon).squeeze() - (1-y1) * torch.log(1 - out[1] + epsilon).squeeze()).mean()
#     loss_y2 = ((-y2 * torch.log(out[2] + epsilon).squeeze() - (1-y2) * torch.log(1 - out[2] + epsilon).squeeze()) * y1.squeeze() / (out[1].detach().squeeze()+1e-9)).mean()
#     return loss_pi + loss_y1 + loss_y2 

def criterion_TR(out, t,trg, y, beta=1., epsilon=1e-9,detach=False):
    pi = out[0].detach() if detach else out[0]
    mu = (1-t)*out[1][0] + t*out[1][1] + trg * (t/(pi+epsilon) - (1-t)/(1-pi+epsilon))
    return 0*beta *((mu.squeeze() - y.squeeze())**2).mean()

def criterion_TR_cvr(out, t,trg, y1, y2, beta=1., epsilon=1e-9):
    pi = out[0].detach().squeeze()
    mu1 = ((1-t)*out[1][0] + t*out[1][1]).squeeze()
    mu2 = ((1-t)*out[1][0]*out[2][0] + t*out[1][1]*out[2][1]).squeeze()
    y1,y2,trg,t = y1.squeeze(),y2.squeeze(),trg.squeeze(),t.squeeze()
    trg_term = trg*(t/(pi+epsilon) - (1-t)/(1-pi+epsilon))
    return beta * (( (y2 - mu2)/(mu1 + epsilon) - (y1-mu1)*mu2/(mu1**2+epsilon) - trg_term)**2).mean()

# def criterion_TR_id(out, trg, y1, y2,beta=1., epsilon=1e-9):
#     return beta *  (y1.squeeze()*(y2.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[2].squeeze())**2 / (out[1].detach().squeeze()+1e-9) ).mean()

# def eval(model,test_matrix, t_grid, TargetReg1,TargetReg2):
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data_utils')

    # i/o
    parser.add_argument('--data_dir', type=str, default='/root/test01/research/CausalCVR/dataset/criteo', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='/root/test01/research/CausalCVR/logs/criteo', help='dir to save result')

    # common
    # parser.add_argument('--num_dataset', type=int, default=100, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=600, help='num of epochs to train')

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



    Result = {'MyNet_tr': {}, 'MyNet': {}}
    # model_name = 'Vcnet_tr'
    best_auuc = 0
    best_qini = 0
    best_params = None
    k = 0

    #for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr']:
    
    for num_epoch in [10]:
        for h in [8]:
            # for init_lr in [1e-4,1e-3,1e-2]:
            # for alpha in [0.1,0.5,1]:
            for init_lr in [1e-5]: #1e-6,
                # for tr_init_lr in [1e-4,1e-3,1e-2]:
                for tr_init_lr in [1e-5]:
                    #if h==32 and init_lr==1e-5 and tr_init_lr==1e-5: continue
                    params = f'{h}_{init_lr }_{tr_init_lr}'
                    # Result[params]={}
                    #Result[model_name]=[]
                    k+=1
                    alpha = 0.5
                    beta = 1.
                    for model_name in ['MyNet_tr','MyNet']: #'Dragonnet_tr','Tarnet'
                        cfg_density = [(12, h, 1, 'relu'), (h, h, 1, 'relu')]
                        cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
                        model = CVRNet_Binary(cfg_density, cfg).to(device)
                        # model = MyNet(cfg_density, num_grid, cfg, degree, knots,cfg_backbone=[(h, h, 1, 'relu')]).to(device)
                        model._initialize_weights()
                        # use Target Regularization
                        if model_name == 'MyNet_tr':
                            isTargetReg = 1
                        else:
                            isTargetReg = 0

                        if isTargetReg:
                            TargetReg1 = TR_Binary().to(device)
                            TargetReg2 = TR_Binary().to(device)
                            TargetReg1._initialize_weights()
                            TargetReg2._initialize_weights()

                        init_lr = init_lr  #0.0001
                        alpha = 0.5 #alpha #0.5
                        tr_init_lr = tr_init_lr #0.001
                        beta = 1. #beta #1.

                        cur_save_path = save_path
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
                        model._initialize_weights()

                        # define optimizer
                        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)
                        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=wd)

                        if isTargetReg:
                            tr_optimizer1 = torch.optim.SGD(TargetReg1.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
                            tr_optimizer2 = torch.optim.SGD(TargetReg2.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

                        print('model : ', model_name)
                        for epoch in range(num_epoch):
                            for idx, (inputs,exposure) in enumerate(train_loader):
                                t = inputs[:, 12]
                                x = inputs[:, :12]
                                y1 = inputs[:, 14]
                                y2 = inputs[:, 13]


                                if isTargetReg:
                                    optimizer.zero_grad()
                                    out = model.forward(x)
                                    trg1,trg2 = TargetReg1(x),TargetReg2(x)
                                    loss = criterion_cvr(out, t,y1,y2,alpha=alpha) + 0*criterion_TR(out, t,trg1, y1, beta=beta,detach=True) + criterion_TR_cvr(out, t,trg2, y1, y2, beta=beta)
                                    loss.backward()
                                    optimizer.step()

                                    tr_optimizer1.zero_grad()
                                    tr_optimizer2.zero_grad()
                                    out = model(x)
                                    trg1, trg2 = TargetReg1(x), TargetReg2(x)
                                    
                                    # tr_loss1 = criterion_TR(out, t,trg1, y1, beta=beta,detach=True)
                                    tr_loss2 = criterion_TR_cvr(out, t,trg2, y1, y2, beta=beta)
                                    # tr_loss1.backward(retain_graph=True) 
                                    # tr_optimizer1.step()
                                    tr_loss2.backward() 
                                    tr_optimizer2.step()
                                else:
                                    optimizer.zero_grad()
                                    out = model.forward(x)
                                    loss = criterion_cvr(out, t,y1,y2,alpha=alpha)
                                    loss.backward()
                                    optimizer.step()

                            epoch += 1
                            if epoch==1 or epoch % verbose == 0 or epoch==num_epoch: #
                                print('current epoch: ', epoch)
                                print('current loss: ', loss.data)
                                if epoch==1 or epoch % 5 == 0 or epoch==num_epoch:
                                    if isTargetReg:
                                        auuc1, auuc2,qini1,qini2 = eval_binary_2(model,test_matrix, TargetReg1,TargetReg2)
                                    else:
                                        auuc1, auuc2,qini1,qini2 = eval_binary_2(model, test_matrix)
                                    
                                    params1 = f'{epoch}_' + params 
                                    Result[model_name][params1] = [auuc1,auuc2,qini1,qini2]
                                    if 'tr' not in model_name:
                                        res = Result[model_name][params1]
                                        res_tr = Result[model_name+'_tr'][params1]
                                        #if res[0]>0 and res[1]>0 and res[2]>0 and res[3]>0 and res_tr[0]>res[0] and res_tr[1]>res[1] and res_tr[2]>res[2] and res_tr[3]>res[3] and (2*res_tr[1]+res_tr[3])>(2*best_auuc+best_qini):
                                        #if res[0]>0 and res[1]>0 and res_tr[0]>res[0] and res_tr[1]>res[1]:
                                        if res[1]>0 and res[3]>0 and res_tr[1]>res[1] and res_tr[3]>res[3]:
                                            ckpt_dir = os.path.join(cur_save_path, "checkpoints")
                                            os.makedirs(ckpt_dir, exist_ok=True)
                                            ckpt_name = f"test_{model_name}_ckpt_{params1}_{'_'.join([f'{r:.4f}' for r in res])}.pth.tar"
                                            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                                            print('-----------------------------------------------------------------')
                                            save_checkpoint({
                                                'model': model_name,
                                                'res': res,
                                                'model_state_dict': model.state_dict(),
                                                'TR_state_dict': [TargetReg1.state_dict(), TargetReg2.state_dict()] if isTargetReg else None
                                            }, model_name=model_name, checkpoint_dir=ckpt_path)
                                            print('-----------------------------------------------------------------')
                                            if res_tr[1]>best_auuc:
                                                best_params = params1 
                                                best_auuc,best_qini = res_tr[1],res_tr[3]
                                    
                                    else:
                                        res = Result[model_name][params1]
                                        if res[1]>0 and res[3]>0:
                                            ckpt_dir = os.path.join(cur_save_path, "checkpoints")
                                            os.makedirs(ckpt_dir, exist_ok=True)
                                            ckpt_name = f"test_{model_name}_ckpt_{params1}_{'_'.join([f'{r:.4f}' for r in res])}.pth.tar"
                                            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                                            print('-----------------------------------------------------------------')
                                            save_checkpoint({
                                                'model': model_name,
                                                'res': res,
                                                'model_state_dict': model.state_dict(),
                                                'TR_state_dict': [TargetReg1.state_dict(), TargetReg2.state_dict()] if isTargetReg else None
                                            }, model_name=model_name, checkpoint_dir=ckpt_path)
                                            print('-----------------------------------------------------------------')


                                    print('current auuc: ', [auuc1,auuc2],' current qini: ',[qini1,qini2],'best res: ', [best_auuc,best_qini],' best_params: ',best_params)
                                    with open(save_path + '/result_test_2.json', 'w') as fp:
                                        json.dump(Result, fp)

                    