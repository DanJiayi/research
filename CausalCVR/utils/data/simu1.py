import torch

def x_t(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    t = (10. * torch.sin(max(x1, x2, x3)) + max(x3, x4, x5)**3)/(1. + (x1 + x5)**2) + \
        torch.sin(0.5 * x3) * (1. + torch.exp(x4 - 0.5 * x3)) + x3**2 + 2. * torch.sin(x4) + 2.*x5 - 6.5
    return t

def x_t_2(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x8 = x[7]
    t = (10. * torch.sin(max(x1, x2, x3)) + max(x3, x4, x5, x8)**3)/(1. + (x1 + x5)**2) + \
        torch.sin(0.5 * x3) * (1. + torch.exp(x4 + x8 - 0.5 * x3)) + x3**2 + 2. * torch.sin(x4) + torch.cos(x8) + 2.*x5 - 6.5
    return t

def x_t_link(t):
    return 1. / (1. + torch.exp(-1. * t))

def t_x_y(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    y = torch.cos((t-0.5) * 3.14159 * 2.) * (t**2 + (4.*max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
    return y

def t_x_y_2(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    y1 = torch.cos((t-0.5) * 3.14159 * 2.) * (t**2 + (4.*max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
    y2 = torch.sin((-0.5*t+0.1) * 3.14159 * 2.) * (-t**2 + t +(4.*max(x1, x7)**3)/(0.5 + x3**3 - x3)*torch.sin(x8))
    return y1,y2

def simu_data1(n_train, n_test):
    train_matrix = torch.zeros(n_train, 8)
    test_matrix = torch.zeros(n_test, 8)
    for _ in range(n_train):
        x = torch.rand(6)
        train_matrix[_, 1:7] = x
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        train_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, -1] = y

    for _ in range(n_test):
        x = torch.rand(6)
        test_matrix[_, 1:7] = x
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        test_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        test_matrix[_, -1] = y

    t_grid = torch.zeros(2, n_test)
    t_grid[0, :] = test_matrix[:, 0].squeeze()

    for i in range(n_test):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_test):
            x = test_matrix[j, 1:7]
            psi += t_x_y(t, x)
        psi /= n_test
        t_grid[1, i] = psi

    return train_matrix, test_matrix, t_grid


def simu_data2(n_train, n_test):
    train_matrix = torch.zeros(n_train, 12)
    test_matrix = torch.zeros(n_test, 12)
    for _ in range(n_train):
        x = torch.rand(8)
        train_matrix[_, 1:9] = x
        t = x_t_2(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        train_matrix[_, 0] = t

        y1,y2 = t_x_y_2(t, x)
        y1 = 0.7*x_t_link(y1 + torch.randn(1)[0] * 0.5)
        y2 = 0.7*x_t_link(y2 + torch.randn(1)[0] * 0.5)

        train_matrix[_, -3] = y1
        train_matrix[_, -2] = y2
        train_matrix[_, -1] = y1 * y2


    for _ in range(n_test):
        x = torch.rand(8)
        test_matrix[_, 1:9] = x
        t = x_t_2(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        test_matrix[_, 0] = t

        y1,y2 = t_x_y_2(t, x)
        y1 = 0.7*x_t_link(y1 + torch.randn(1)[0] * 0.5)
        y2 = 0.7*x_t_link(y2 + torch.randn(1)[0] * 0.5)

        test_matrix[_, -3] = y1
        test_matrix[_, -2] = y2
        test_matrix[_, -1] = y1 * y2

    t_grid = torch.zeros(3, n_test)
    t_grid[0, :] = test_matrix[:, 0].squeeze()

    for i in range(n_test):
        psi1,psi2 = 0,0
        t = t_grid[0, i]
        for j in range(n_test):
            x = test_matrix[j, 1:9]
            y1,y2 = t_x_y_2(t, x)
            y1 = 0.7 * x_t_link(y1)
            y2 = 0.7 * x_t_link(y2)
            psi1 += y1
            psi2 += y2
        psi1 /= n_test
        psi2 /= n_test
        t_grid[1, i] = psi1
        t_grid[2, i] = psi2

    return train_matrix, test_matrix, t_grid


if __name__ == '__main__':
    train_matrix, test_matrix, t_grid = simu_data2(500, 200)
    print(train_matrix.shape,test_matrix.shape,t_grid.shape)
    y1,y2,y3 = train_matrix[:,-3],train_matrix[:,-2],train_matrix[:,-1]
    print(y1.mean(),y2.mean(),y3.mean())
    y1,y2,y3 = test_matrix[:,-3],test_matrix[:,-2],test_matrix[:,-1]
    print(y1.mean(),y2.mean(),y3.mean())
    print(t_grid[0,:].mean(),t_grid[1,:].mean(),t_grid[2,:].mean())

