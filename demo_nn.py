#%%
import torch
import numpy as np

#%%

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(data)


#%%
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(tensor)
#%%

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(tensor)

#%%
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F

#%%
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 100
# Create the training dataset and dataloader
train_dataset = torchvision.datasets.MNIST('./content', train=True, download=True, transform=transform)

#%%
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#%%
data = iter(train_dataloader)
samples,labels=next(data)
print(f"number of samples {samples.shape}")
print(f"number of labels {labels.shape}")

#%%
plt.figure(figsize=(10,8))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(samples[i][0],cmap='BuPu')
plt.savefig('numbers.png')

#%%
# Definir una operacion lineal en toch
m = nn.Linear(20, 50)

input = torch.randn(512, 20)
print(input)
print("input shape: ", input.shape)
output = m(input)
print("output shape: ", output.shape)


#%%
import torch.nn.functional as F
print("samples.shape",samples.shape)
x= samples.reshape(-1,28*28)
print("x.shape",x.shape)

# definimos la operacion
m1 = nn.Linear(28*28, 20)

# ------------- Aplicamos -------
# aplicamos 1 capa de la red neuronal

# aplicamos la Op lineal
x= m1(x)
# aplico op nolineal
x = F.relu(x)
print("m1 output",x.shape)

# ----------------------------

#%%

# Red de una capa oculta
class REDN(nn.Module):
    def __init__(self, input_size=28*28,):
        super(REDN, self).__init__()
        # capa de entrada 40 neuronas
        self.f1 = nn.Linear(input_size, 40)
        # capa oculta 20 neuronal
        self.f2 = nn.Linear(40, 20)
        self.f3 = nn.Linear(20,10)
    def forward(self, x):
        print("x input: ", x.shape)
        # aplicamos capa entrada
        x = self.f1(x)
        x = F.relu(x)
        print("f1 shape: ", x.shape)
        # aplicamos capa oculta
        x = self.f2(x)
        x = F.relu(x)
        print("f2 shape: ", x.shape)
        # aplicamos salida
        out = self.f3(x)
        print("out shape: ", out.shape)
        return out

model = REDN(input_size=28*28)
print(model)


#%%
data = iter(train_dataloader)
samples,labels=next(data)
print(f"number of samples {samples.shape}")
print(f"number of labels {labels.shape}")

# paso forward
x = samples.reshape(-1,28*28)

#%%
y = model(x)

print(" y shape", y.shape)

# ------ 1 EPOCH -----------------
for i, (samples,labels) in enumerate(train_dataloader):
    x=samples.reshape(-1,28*28)
    y = model(x)
    print("batch ", i)


#%%
