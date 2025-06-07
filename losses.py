import torch

def pde_loss(X, T_hat, alpha, criterion):
    # PDE loss
    dT_dX = torch.autograd.grad(T_hat, X, torch.ones_like(T_hat), create_graph=True)[0]
    dT_dt = dT_dX[:, 1]
    dT_dx = dT_dX[:, 0]
    
    # Laplacian terms
    dT_dxx = torch.autograd.grad(dT_dx, X, torch.ones_like(dT_dx), create_graph=True)[0][:, 0]
    
    return criterion(alpha * (dT_dxx), dT_dt + (T_hat[:,0] * dT_dx)) #viscous wird als 0.1/sinpi gewählt