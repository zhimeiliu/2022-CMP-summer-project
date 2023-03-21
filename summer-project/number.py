import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def train_one_epoch():
    # 5. Train the model
    totalloss=0
    num=0
    for inputs, target in train_loader:
        target = torch.eye(10)[target]
        output = net(inputs)
 
        loss = criterion(output, target)
        totalloss+=loss
        num+=1
        if num%50==0:
            average=totalloss/num/128*10000
            print(f'[{num*128}/60000 ({round(num*128/600)}%)]    AverageLoss:{round(average.item(),4)}')

        net.zero_grad()
        loss.backward()
        optimizer.step()
    averageloss = totalloss / len(train_dataset) *10000   
    print(f'average loss {round(averageloss.item(),4)}')
    return averageloss.item()


@torch.no_grad()
def validate_one_epoch():
    """Validate one epoch."""
    correct = 0.
    testloss= 0.
    net.eval()
    for inputs, target in val_loader:
        output = net(inputs)  
        _, pred = output.max(1)
        correct += (pred == target).sum()
        target = torch.eye(10)[target]
        testloss += criterion(output, target).item()
    accuracy = correct / len(val_dataset) * 100.
    testloss /= len(val_dataset) /10000
    print(f'{accuracy:.2f}% correct')
    return testloss, accuracy

def train_n_epoch(n):
    df = pd.DataFrame(columns=['epoch', 'train loss', 'test loss', 'accuracy'])
    for i in range(n):
        print('-----------------------')
        print(f'epoch {i+1}')
        a=train_one_epoch()
        [b, c] =validate_one_epoch()
        df.loc[i] = [i+1] + [a] + [b] +[f'{c.item()}%']
        torch.save(net.state_dict(),f'epoch__{i+1}.pth')   
    print(df)
    df.to_csv('16*16.csv')

# 1. Build a computation graph
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(9216, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output
net = Net()

# 2. Define the optimizer
optimizer = optim.Adadelta(net.parameters(), lr=1.)  # 2. Setup optimizer


# criterion = nn.NLLLoss()  # 3. Setup criterion
criterion = nn.BCELoss()


# 4. Setup data
transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(
    'data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
val_dataset = datasets.MNIST(
    'data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128)

# net.load_state_dict(torch.load('epoch1.pth'))

train_n_epoch(30)

