import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from data import ADNI_MRI
from network import MRICNN
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_step(model, optim, dataloader, loss_fn, device):
    print("HERE 1")
    model.train()
    total_loss = 0
    for (X, y) in  dataloader:
        print(X.size())
        X = X.to(device)
        y = y.to(device)

        optim.zero_grad()
        pred = model(X.squeeze())
        loss = loss_fn(pred, y)
        total_loss += loss
        optim.step()

    print("HERE 2")
    return total_loss / len(dataloader)

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    for batch, (X, y) in  enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X.squeeze())
        loss = loss_fn(pred, y)
        total_loss += loss

    return total_loss / len(dataloader)

root = '/home/jacob/Documents/data/T1-5mm-AXIAL/ADNI'
csvpath = '/home/jacob/Documents/data/cleaned_adni.csv'

dataset = ADNI_MRI(dataroot=root, csvpath=csvpath)
train_set, test_set = random_split(dataset, lengths=[1600, 51])
train_set, valid_set = random_split(train_set, lengths=[1280, 320])


train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)

device = 'cpu'
if torch.cuda.is_available(): 
    print("Using cuda device")
    device = 'cuda'

model = MRICNN()
model.to(device)
optim = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 20

train_losses, val_losses = [], []
for epoch in range(epochs):
    print(f"Starting epoch {epoch}...............")
    train_loss = train_step(model, optim, train_dataloader, loss_fn, device)
    val_loss = test_step(model, train_dataloader, loss_fn, device)

    train_losses.append(train_loss)
    val_losses.append(val_losses)

    print(f'Train loss: {train_loss}\Val loss : {val_loss}')

plt.figure()
plt.plot(train_losses, label='Train loss')
plt.plot(val_loss, label='Val loss')    
plt.savefig('test.png')

print("done!")


