import torch
import torch.nn as nn
from utility import weights_init
import torch.nn.functional as F
import numpy as np

#1 torch.Size([64, 1, 30, 133])
#2 torch.Size([64, 20, 1, 133])
#3 torch.Size([64, 2660])
import nvidia_smi

def print_free_memory(point):
    print(point, "%.3f"%((11178*(1024**2) - torch.cuda.memory_allocated())/1024**3) , "GB")


    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    #print('gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    print (res.memory)

class beta(nn.Module):
    def __init__(self, num_clusters=0):
        super(beta, self).__init__()
        
        self.num_clusters = num_clusters

        print (num_clusters)
    
        self.conv1 = nn.Conv2d( 1, 50, kernel_size=(5, 1), padding=0)
        self.conv2 = nn.Conv2d(50, 40, kernel_size=(5, 1), padding=0)
        self.conv3 = nn.Conv2d(40, 20, kernel_size=(4, 1), padding=0)

        self.fc1 = nn.Linear(2660, (num_clusters+1)*400)
        self.fc2 = nn.Linear( (num_clusters+1)*400, 17)

        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):

        #print_free_memory("before initializing")

        #print_free_memory("after initializing")

        #print ('first', x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x), inplace = True), (2, 1), padding=0)
        #print ('after conv 1', x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x), inplace = True), (2, 1), padding=0)
        #print ('after conv 2', x.shape)
        #print ('after conv 2', x.shape)
        x = F.relu(self.conv3(x), inplace = True)
        #print ('after conv 3', x.shape)

        #print ('after conv 3', x.shape)

        x = x.view(-1, np.prod(x.shape[1:]))       

        x = torch.tanh(x)

        y = F.relu(self.fc1(x), inplace = True)

        #print ('after fc 1', y.shape)

        #print ('after fc 1', y.shape)
        y = self.fc2(y)
        #print ('after fc 2', y.shape)
        #print ('after fc 2', y.shape)
        #exit()
        
        return x, F.log_softmax(y, dim=1)


'''
class beta(nn.Module):
    def __init__(self):
        super(beta, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(5, 1), padding=0)
        self.conv2 = nn.Conv2d(50, 40, kernel_size=(5, 1), padding=0)
        self.conv3 = nn.Conv2d(40, 20, kernel_size=(4, 1), padding=0)

        self.fc1 = nn.Linear(2660, 400)
        self.fc2 = nn.Linear(400, 17)

        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 1), padding=0)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 1), padding=0)
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 2660)
        
        x = torch.tanh(x)

        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        
        return x, F.log_softmax(y, dim=1)
'''


class alpha (nn.Module):
    def __init__(self, num_clusters):
        super(alpha, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, kernel_size=(5, 1), padding=0)
        self.conv2 = nn.Conv2d(50, 40, kernel_size=(5, 1), padding=0)
        self.conv3 = nn.Conv2d(40, 20, kernel_size=(4, 1), padding=0)

        self.fc1 = nn.Linear(2660, 400)
        self.fc2 = nn.Linear(400, num_clusters)

        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 1), padding=0)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 1), padding=0)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 2660)
        x = torch.tanh(x)

        y = F.relu(self.fc1(x))
        y = self.fc2(y)

        return x, F.log_softmax(y, dim=1)
