import torch

def generate_burger_equation_dataset(nx, nt):
    # Geo
    x = torch.linspace(-1, 1, nx) #tensor spaced between 0 and 1 with nx elements
    t = torch.linspace(0, 1, nt)
    X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T.requires_grad_(True)
    # reshape the stacked tensor into a shape mit erster dim 2

    # IC & BC
    X_BC = X[(X[:, 0] == x[0]) | (X[:, 0] == x[-1]) ] #periodic boundary condition
    X_IC = X[X[:, 1] == t[0]] 
    X_train = torch.cat([X_BC, X_IC])  #concatenate all the points for training
    T_BC = torch.zeros(len(X_BC)).view(-1, 1) # u(-1,t) = u(1,t) = 0 
    T_IC = torch.sin(- torch.pi * X_IC[:, 0]).view(-1, 1) #u(x,0) = - sin(pi*x) 
    T_train = torch.cat([T_BC, T_IC])

    return X, X_train, T_train
