import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import PINN
from dataset import generate_burger_equation_dataset
from losses import pde_loss
from trainer import pinn_trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)

# Hyperparameters
alpha = 0.01 / np.pi # viscosity parameter
nx = 200
nt = 200

# Generate dataset
X, X_train, T_train = generate_burger_equation_dataset(nx, nt)
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of T_train:", T_train.shape)

# Create PINN model
model = PINN(2, 64, 1, 6).to(device) # 2 input, 64 neurons, 6 layers 1 output

# Define optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=1e-3) # choose learning rate dynamically - hyperparameter, others are SGD (learning rate is fixed, so lr should be chosen small)
criterion = nn.MSELoss()

# Train the PINN model
pinn_trainer(1000, model, optimizer, criterion, X, X_train, T_train, device, alpha)

# Plotting
with torch.no_grad():
    y_pred = model(X.to(device))
    y_pred = y_pred.reshape(nx, nt).cpu().numpy()

plt.imshow(y_pred, extent=[0,1,-1,1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label="u(x, t)")
plt.xlabel("Time (t)")
plt.ylabel("Space (x)")
plt.title("Solution of Burger Equation")
plt.show()


