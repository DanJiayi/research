import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

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
            t_grid_hat[1, _], t_grid_hat[2, _]= out2
        device = t_grid_hat.device
        t_grid_hat = t_grid_hat.to(device)
        t_grid = t_grid.to(device)
        mse1 = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        mse2 = ((t_grid_hat[2, :].squeeze() - t_grid[2, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse1, mse2