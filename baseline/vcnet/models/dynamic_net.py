import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import wasserstein

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data_utils is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out.cuda() # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out

class Vcnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots):
        super(Vcnet, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'elu':
                density_blocks.append(nn.ELU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        #t_hidden = torch.cat((torch.unsqueeze(t, 1), x), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


"""
cfg_density = [(3,4,1,'relu'), (4,6,1,'relu')]
num_grid = 10
cfg = [(6,4,1,'relu'), (4,1,1,'id')]
degree = 2
knots = [0.2,0.4,0.6,0.8]
D = Dynamic_net(cfg_density, num_grid, cfg, degree, knots)
D._initialize_weights()
x = torch.rand(10, 3)
t = torch.rand(10)
y = torch.rand(10)
out = D.forward(t,x)
"""

# Targeted Regularizer

class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        #self.weight.data_utils.normal_(0, 0.01)
        self.weight.data.zero_()

# ------------------------------------------ Drnet and Tarnet ------------------------------------------- #

class Treat_Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', istreat=1, isbias=1, islastlayer=0):
        super(Treat_Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias
        self.istreat = istreat
        self.islastlayer = islastlayer

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if self.istreat:
            self.treat_weight = nn.Parameter(torch.rand(1, self.outd), requires_grad=True)
        else:
            self.treat_weight = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, [0]]

        out = torch.matmul(x_feature, self.weight)

        if self.istreat:
            out = out + torch.matmul(x_treat, self.treat_weight)
        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)

        return out

class Multi_head(nn.Module):
    def __init__(self, cfg, isenhance):
        super(Multi_head, self).__init__()

        self.cfg = cfg # cfg does NOT include the extra dimension of concat treatment
        self.isenhance = isenhance  # set 1 to concat treatment every layer/ 0: only concat on first layer

        # we default set num of heads = 5
        # self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]
        self.pt = [0.0, 0.6, 0.7, 0.8, 0.9, 1.]

        self.outdim = -1
        # construct the predicting networks
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                self.outdim = layer_cfg[1]
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                           islastlayer=0))
        blocks.append(last_layer)
        self.Q1 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q2 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q3 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q4 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q5 = nn.Sequential(*blocks)

    def forward(self, x):
        # x = [treatment, features]
        out = torch.zeros(x.shape[0], self.outdim).cuda()
        t = x[:, 0]

        idx1 = list(set(list(torch.where(t >= self.pt[0])[0].cpu().numpy())) & set(torch.where(t < self.pt[1])[0].cpu().numpy()))
        idx2 = list(set(list(torch.where(t >= self.pt[1])[0].cpu().numpy())) & set(torch.where(t < self.pt[2])[0].cpu().numpy()))
        idx3 = list(set(list(torch.where(t >= self.pt[2])[0].cpu().numpy())) & set(torch.where(t < self.pt[3])[0].cpu().numpy()))
        idx4 = list(set(list(torch.where(t >= self.pt[3])[0].cpu().numpy())) & set(torch.where(t < self.pt[4])[0].cpu().numpy()))
        idx5 = list(set(list(torch.where(t >= self.pt[4])[0].cpu().numpy())) & set(torch.where(t <= self.pt[5])[0].cpu().numpy()))

        idx1 = torch.tensor(idx1).cuda()
        idx2 = torch.tensor(idx2).cuda()
        idx3 = torch.tensor(idx3).cuda()
        idx4 = torch.tensor(idx4).cuda()
        idx5 = torch.tensor(idx5).cuda()

        if len(idx1)>0:
            out1 = self.Q1(x[idx1, :])
            out[idx1, :] = out[idx1, :] + out1

        if len(idx2)>0:
            out2 = self.Q2(x[idx2, :])
            out[idx2, :] = out[idx2, :] + out2

        if len(idx3)>0:
            out3 = self.Q3(x[idx3, :])
            out[idx3, :] = out[idx3, :] + out3

        if len(idx4)>0:
            out4 = self.Q4(x[idx4, :])
            out[idx4, :] = out[idx4, :] + out4

        if len(idx5)>0:
            out5 = self.Q5(x[idx5, :])
            out[idx5, :] = out[idx5, :] + out5

        return out


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        # 线性变换 A s^{wi}_{it} + b
        self.fc = nn.Linear(input_dim, hidden_dim)
        # 潜在向量 u
        self.u = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, rnn_output):
        # 应用线性变换和 tanh 激活
        transformed = torch.tanh(self.fc(rnn_output))
        # 计算相似度得分 s^{wi}_{it}^T u
        similarity = torch.matmul(transformed, self.u)
        # 对相似度得分进行 softmax 归一化以得到权重 α_{it}
        weights = F.softmax(similarity.squeeze(-1), dim=1).unsqueeze(-1)
        # 计算加权平均
        context_vector = torch.sum(weights * transformed, dim=1)
        return context_vector, weights


class Treat_LTEE(nn.Module):
    def __init__(self, hidden_dim, cfg_density):
        super(Treat_LTEE, self).__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.attention = Attention(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.treat = nn.Linear(in_features=1, out_features=hidden_dim, bias=1)

        density_blocks = []
        for layer_idx, layer_cfg in enumerate(cfg_density):
            density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'elu':
                density_blocks.append(nn.ELU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())

        self.short_term_mlp = nn.Sequential(*density_blocks)

        density_blocks = []
        for layer_idx, layer_cfg in enumerate(cfg_density):
            density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'elu':
                density_blocks.append(nn.ELU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())

        self.long_term_mlp = nn.Sequential(*density_blocks)


    def forward(self, x):
        t, x = x[:, 0], x[:, 1:]
        treat = self.treat(t.unsqueeze(dim=-1))

        rnn_output, _ = self.rnn(x.unsqueeze(dim=1).repeat(1, 7, 1))

        short_term_output = self.short_term_mlp(torch.cat((rnn_output, treat.unsqueeze(dim=1).repeat(1, 7, 1)), dim=2))
        long_term_input, _ = self.attention(rnn_output)
        long_term_output = self.long_term_mlp(torch.cat((long_term_input, treat), dim=1))

        # short_term_output = self.short_term_mlp(rnn_output)
        # long_term_input, _ = self.attention(rnn_output)
        # long_term_output = self.long_term_mlp(long_term_input)

        return rnn_output, short_term_output, long_term_output


class Multi_head_ltee(nn.Module):
    def __init__(self, cfg, isenhance, pt, hidden_size):
        super(Multi_head_ltee, self).__init__()

        self.cfg = cfg # cfg does NOT include the extra dimension of concat treatment
        self.isenhance = isenhance  # set 1 to concat treatment every layer/ 0: only concat on first layer

        # we default set num of heads = 5
        # self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]
        self.pt = pt

        self.outdim = -1
        # construct the predicting networks
        self.Q1 = Treat_LTEE(hidden_dim=hidden_size, cfg_density=cfg)
        self.Q2 = Treat_LTEE(hidden_dim=hidden_size, cfg_density=cfg)
        self.Q3 = Treat_LTEE(hidden_dim=hidden_size, cfg_density=cfg)
        self.Q4 = Treat_LTEE(hidden_dim=hidden_size, cfg_density=cfg)
        self.Q5 = Treat_LTEE(hidden_dim=hidden_size, cfg_density=cfg)


    def forward(self, x):
        # x = [treatment, features]
        rep_s = torch.zeros(x.shape[0], 7, 50).cuda()
        out_s = torch.zeros(x.shape[0], 7).cuda()
        out_y = torch.zeros(x.shape[0], 1).cuda()
        t = x[:, 0]

        idx1 = list(set(list(torch.where(t >= self.pt[0])[0].cpu().numpy())) & set(torch.where(t < self.pt[1])[0].cpu().numpy()))
        idx2 = list(set(list(torch.where(t >= self.pt[1])[0].cpu().numpy())) & set(torch.where(t < self.pt[2])[0].cpu().numpy()))
        idx3 = list(set(list(torch.where(t >= self.pt[2])[0].cpu().numpy())) & set(torch.where(t < self.pt[3])[0].cpu().numpy()))
        idx4 = list(set(list(torch.where(t >= self.pt[3])[0].cpu().numpy())) & set(torch.where(t < self.pt[4])[0].cpu().numpy()))
        idx5 = list(set(list(torch.where(t >= self.pt[4])[0].cpu().numpy())) & set(torch.where(t <= self.pt[5])[0].cpu().numpy()))

        idx1 = torch.tensor(idx1).cuda()
        idx2 = torch.tensor(idx2).cuda()
        idx3 = torch.tensor(idx3).cuda()
        idx4 = torch.tensor(idx4).cuda()
        idx5 = torch.tensor(idx5).cuda()

        if len(idx1)>0:
            rep1_s, out1_s, out1_y = self.Q1(x[idx1, :])
            rep_s[idx1, :] = rep_s[idx1, :] + rep1_s
            out_s[idx1, :] = out_s[idx1, :] + out1_s.squeeze(dim=-1)
            out_y[idx1, :] = out_y[idx1, :] + out1_y

        if len(idx2)>0:
            rep2_s, out2_s, out2_y = self.Q2(x[idx2, :])
            rep_s[idx2, :] = rep_s[idx2, :] + rep2_s
            out_s[idx2, :] = out_s[idx2, :] + out2_s.squeeze(dim=-1)
            out_y[idx2, :] = out_y[idx2, :] + out2_y

        if len(idx3)>0:
            rep3_s, out3_s, out3_y = self.Q3(x[idx3, :])
            rep_s[idx3, :] = rep_s[idx3, :] + rep3_s
            out_s[idx3, :] = out_s[idx3, :] + out3_s.squeeze(dim=-1)
            out_y[idx3, :] = out_y[idx3, :] + out3_y

        if len(idx4)>0:
            rep4_s, out4_s, out4_y = self.Q4(x[idx4, :])
            rep_s[idx4, :] = rep_s[idx4, :] + rep4_s
            out_s[idx4, :] = out_s[idx4, :] + out4_s.squeeze(dim=-1)
            out_y[idx4, :] = out_y[idx4, :] + out4_y

        if len(idx5)>0:
            rep5_s, out5_s, out5_y = self.Q5(x[idx5, :])
            rep_s[idx5, :, :] = rep_s[idx5, :, :] + rep5_s
            out_s[idx5, :] = out_s[idx5, :] + out5_s.squeeze(dim=-1)
            out_y[idx5, :] = out_y[idx5, :] + out5_y

        wass = 0
        for i in range(rep_s.shape[1]):
            dist1 = torch.cat((rep_s[:, i, :], t.unsqueeze(dim=1)), axis=1)
            shuffle_idx = torch.randperm(t.shape[0])
            dist2 = torch.cat((rep_s[:, i, :], t[shuffle_idx].unsqueeze(dim=1)), axis=1)
            wass_t, _ = wasserstein(dist1, dist2)
            wass += wass_t

        return wass, out_s, out_y



class Drnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(Drnet, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'elu':
                density_blocks.append(nn.ELU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # multi-head outputs blocks
        self.Q = Multi_head_ltee(cfg, isenhance)
        self.Q.cuda()

    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(0, 1.)  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()



class LTEE_Drnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance, pt):
        super(LTEE_Drnet, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'elu':
                density_blocks.append(nn.ELU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # multi-head outputs blocks
        self.Q = Multi_head_ltee(cfg, isenhance, pt, hidden_size=cfg_density[-1][1])
        self.Q.cuda()


    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        g = self.density_estimator_head(t, hidden)
        wass, out_s, out_y = self.Q(t_hidden)

        return g, wass, out_s, out_y

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(0, 1.)  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


"""
cfg_density = [(3,4,1,'relu'), (4,6,1,'relu')]
cfg = [(6,4,1,'relu'), (4,1,1,'id')]
num_grid = 10
isenhance = 1
D = Drnet(cfg_density, num_grid, cfg, isenhance)
D._initialize_weights()
x = torch.rand(10, 3)
t = torch.rand(10)
y = torch.rand(10)
out = D.forward(t, x)
"""