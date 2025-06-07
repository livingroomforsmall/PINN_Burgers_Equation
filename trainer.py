import torch
from losses import pde_loss
def pinn_trainer(epochs, model, optimizer, criterion, X, X_train, T_train, device, alpha):
    model.train()
    X, X_train, T_train = X.to(device), X_train.to(device), T_train.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        Tt_hat = model(X_train)
        loss_train = criterion(T_train, Tt_hat)

        T_hat = model(X)

        loss_physics = pde_loss(X, T_hat, alpha, criterion=criterion)
        loss = loss_physics + loss_train

        loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')