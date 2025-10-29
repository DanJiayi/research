import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from sklift.metrics import uplift_auc_score, qini_auc_score
from causalml.metrics import auuc_score, qini_score

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def test_ltee_drnet(model, x, t, y, cfT, cfY, targetreg=None, norm=False, mu=0., std=1.):
    """test factual and counterfactual performance on test/valid dataset"""
    # check factual performance
    if targetreg:
        out = model.forward(t, x)
        tr_out = targetreg(t).data
        g = out[0].data.squeeze()
        pred_y = out[1].data.squeeze() + tr_out / (g + 1e-6)
    else:
        g, wass, out_s, out_y = model.forward(t, x)

    loss_y = ((out_y - y.view(-1, 1)) ** 2).mean().item()

    # check counterfactual performance
    loss_cfy_list = []
    for i in range(cfT.shape[0]):
        cft, cfy = cfT[i, :], cfY[i, :]
        if targetreg:
            out = model.forward(cft, x)
            tr_out = targetreg(cft).data
            g = out[0].data.squeeze()
            pred_cfy = out[1].data.squeeze() + tr_out / (g + 1e-6)
        else:
            g, wass, out_s, cf_out_y = model.forward(cft, x)

        loss_cfy_list.append(((cf_out_y - cfy.view(-1, 1)) ** 2).mean().item())

    loss_cfy = np.array(loss_cfy_list).mean()

    return loss_y, loss_cfy

def test(model, x, t, y, cfT, cfY, targetreg=None, norm=False, mu=0., std=1.):
    """test factual and counterfactual performance on test/valid dataset"""
    # check factual performance
    if targetreg:
        out = model.forward(t, x)
        tr_out = targetreg(t).data
        g = out[0].data.squeeze()
        pred_y = out[1].data.squeeze() + tr_out / (g + 1e-6)
    else:
        out = model.forward(t, x)
        pred_y = out[1].data.squeeze()

    if norm:
        y = y * std + mu
        pred_y = pred_y * std + mu
    loss_y = ((pred_y - y) ** 2).mean().item()

    # check counterfactual performance
    loss_cfy_list = []
    for i in range(cfT.shape[0]):
        cft, cfy = cfT[i, :], cfY[i, :]
        if targetreg:
            out = model.forward(cft, x)
            tr_out = targetreg(cft).data
            g = out[0].data.squeeze()
            pred_cfy = out[1].data.squeeze() + tr_out / (g + 1e-6)
        else:
            out = model.forward(cft, x)
            pred_cfy = out[1].data.squeeze()

        if norm:
            cfy = cfy * std + mu
            pred_cfy = pred_cfy * std + mu

        loss_cfy_list.append(((pred_cfy - cfy) ** 2).mean().item())
    loss_cfy = np.array(loss_cfy_list).mean()

    return loss_y, loss_cfy

def curve(model, test_matrix, t_grid, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader): # n个样本，都取第一个t
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out

        device = t_grid_hat.device
        t_grid_hat = t_grid_hat.to(device)
        t_grid = t_grid.to(device)
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            out = out.mean()
            t_grid_hat[1, _] = out
        device = t_grid_hat.device
        t_grid_hat = t_grid_hat.to(device)
        t_grid = t_grid.to(device)
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse


def curve_2(model,test_matrix, t_grid, targetreg1=None, targetreg2=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(3, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg1 is None:
        for _ in range(n_test):
            for idx, (inputs,y2) in enumerate(test_loader): # n个样本，都取第一个t
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:-2]
                break
            out = model.forward(t, x)
            out1,out2 = out[1].data.squeeze(),out[2].data.squeeze()
            out1,out2 = out1.mean(),out2.mean()
            t_grid_hat[1, _], t_grid_hat[2, _]= out1,out2

        device = t_grid_hat.device
        t_grid_hat = t_grid_hat.to(device)
        t_grid = t_grid.to(device)
        mse1 = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        mse2 = ((t_grid_hat[2, :].squeeze() - t_grid[2, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse1, mse2
    else:
        for _ in range(n_test):
            for idx, (inputs,y2) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:-2]
                break
            out = model.forward(t, x)
            tr_out1,tr_out2 = targetreg1(t).data, targetreg2(t).data
            g = out[0].data.squeeze()
            out1,out2 = out[1].data.squeeze() + tr_out1 / (g + 1e-6),out[2].data.squeeze() + tr_out2 / (g + 1e-6)
            out1,out2 = out1.mean(),out2.mean()
            t_grid_hat[1, _], t_grid_hat[2, _]= out1,out2
        device = t_grid_hat.device
        t_grid_hat = t_grid_hat.to(device)
        t_grid = t_grid.to(device)
        mse1 = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        mse2 = ((t_grid_hat[2, :].squeeze() - t_grid[2, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse1, mse2

def eval_binary_2(model, test_matrix, targetreg1=None, targetreg2=None, batch_size=1024):
    def h(t, pi):
        t = t.view(-1, 1)       # 确保列向量
        pi = pi.view(-1, 1)     # 同形状
        return t / (pi + 1e-9) - (1 - t) / (1 - pi + 1e-9)

    was_training = model.training
    model.eval()

    preds1, preds2 = [], []
    ys1, ys2, ts = [], [], []

    test_loader = get_iter(test_matrix, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for idx, (inputs, exposure) in enumerate(test_loader):
            t = inputs[:, 12]
            x = inputs[:, :12]
            y1 = inputs[:, 14]
            y2 = inputs[:, 13]

            out = model(x)

            if targetreg1 is None:
                pred1 = torch.flatten(out[1][1] - out[1][0])
                pred2 = torch.flatten(out[2][1] - out[2][0])
            else:
                trg1, trg2 = targetreg1(x), targetreg2(x)
                pred1 = torch.flatten(
                    out[1][1]
                    + trg1 * h(torch.ones_like(t), out[0])
                    - out[1][0]
                    - trg1 * h(torch.zeros_like(t), out[0])
                )
                pred2 = torch.flatten(
                    out[2][1]
                    + trg2 * h(torch.ones_like(t), out[0])
                    - out[2][0]
                    - trg2 * h(torch.zeros_like(t), out[0])
                )

            preds1.append(pred1.detach().cpu())
            preds2.append(pred2.detach().cpu())
            ys1.append(y1.detach().cpu().view(-1))
            ys2.append(y2.detach().cpu().view(-1))
            ts.append(t.detach().cpu().view(-1))

    preds1 = torch.cat(preds1).numpy()
    preds2 = torch.cat(preds2).numpy()
    y1 = torch.cat(ys1).numpy()
    y2 = torch.cat(ys2).numpy()
    t = torch.cat(ts).numpy()

    assert len(preds1) == len(y1) == len(t), \
        f"Inconsistent lengths: preds1={len(preds1)}, y1={len(y1)}, t={len(t)}"

    auuc1 = uplift_auc_score(y1, preds1, t)
    qini1 = qini_auc_score(y1, preds1, t)

    mask = (y1 == 1)


    # # CTR
    # df_ctr = pd.DataFrame({'y': y1, 'treatment': t, 'score': preds1})
    # auuc1 = auuc_score(df_ctr, outcome_col='y', treatment_col='treatment', y_pred_col='score')
    # qini1 = qini_score(df_ctr, outcome_col='y', treatment_col='treatment', y_pred_col='score')

    # # CVR
    # df_cvr = pd.DataFrame({'y': y2[mask], 'treatment': t[mask], 'score': preds2[mask]})
    # auuc2 = auuc_score(df_cvr, outcome_col='y', treatment_col='treatment', y_pred_col='score')
    # qini2 = qini_score(df_cvr, outcome_col='y', treatment_col='treatment', y_pred_col='score')

    auuc2 = uplift_auc_score(y2[mask], preds2[mask], t[mask])
    qini2 = qini_auc_score(y2[mask], preds2[mask], t[mask])

    if was_training:
        model.train()

    return auuc1, auuc2, qini1, qini2


