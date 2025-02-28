import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Batch_Data_Learning
from Batch_Data_Learning.custom_batch_norm import *

def setBatchSize(batchSize):
    Batch_Data_Learning.custom_batch_norm.batchSize = batchSize

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              CustomBatchNormManualModule(n_neurons=out_channels), 
            #   nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True) 
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) 
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True) 
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) 
        self.conv5 = conv_block(512, 1028, pool=True) 
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))  
        
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 1028 x 1 x 1
                                        nn.Flatten(), # 1028 
                                        nn.Linear(1028, num_classes)) # 1028 -> num_classes
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out



resnet18 = models.resnet18(weights=None)
for name, module in resnet18.named_modules():
    if isinstance(resnet18.bn1, nn.BatchNorm2d):
        num_features = resnet18.bn1.num_features
        resnet18.bn1 = CustomBatchNormManualModule(resnet18.bn1.num_features, resnet18.bn1.eps, resnet18.bn1.momentum)

    if isinstance(module, nn.BatchNorm2d):
        parent = dict(resnet18.named_modules())[name.rsplit('.', 1)[0]]  # Get the parent module
        setattr(parent, name.split('.')[-1], CustomBatchNormManualModule(module.num_features, module.eps, module.momentum))  # Replace the layer

resnet18.fc = nn.Linear(resnet18.fc.in_features, 200) 



class CovtypeNN(nn.Module):
    def __init__(self):
        super(CovtypeNN, self).__init__()
        self.fc1 = nn.Linear(54, 128)  
        self.dropout1 = nn.Dropout(0.1) 
        
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.1)
        
        self.output_layer = nn.Linear(64, 2)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        x = self.output_layer(x)
        return x  



class PACSModel(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=7):
        super(PACSModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), 128)  
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)

        x = self.fc2(x)
        
        return x



class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Conv layer 1
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Conv layer 2
        self.pool = nn.MaxPool2d(2)  # Define AvgPool layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer 1 (corrected)
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer 2
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected output layer

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool(x) 
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)  
        x = x.view(x.size(0), -1)  
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)  
        return x
    


batch_learning_model = {
    "mnist": None, # add in later***
    "cifar10": ResNet9(3,10),
    "cifar100": ResNet9(3,10),
    "tinyimagenet": resnet18,
    "covtype": CovtypeNN(),
    "pacs": PACSModel(input_shape=(3, 227, 227),num_classes=7),
    "digitdg": LeNet(10)
}