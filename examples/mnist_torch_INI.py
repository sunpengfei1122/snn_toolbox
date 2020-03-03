#AlexNet & MNIST


import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import numpy as np

from torch.autograd import Variable
import torchvision


from keras.utils import np_utils
import os
#from snntoolbox.pytorch2keras.converter import pytorch_to_keras

from keras.datasets import mnist
import keras
from keras import backend as K
K.set_image_data_format('channels_first')  
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser 
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp'))
os.makedirs(path_wd, exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input so we can train ANN with it.
# Will be converted back to integers for SNN layer.
x_train = x_train / 255
x_test = x_test / 255

# Add a channel dimension.
axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
x_train = np.expand_dims(x_train, axis)
x_test = np.expand_dims(x_test, axis)

# One-hot encode target vectors.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
# SNN toolbox will not do any training, but we save a subset of the training
# set so the toolbox can use it when normalizing the network parameters.
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])


from keras import metrics

#Alexnet, test the conv, flatten, fc, pooling, bn layers. 
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #AlexCONV1(3,96, k=11,s=4,p=0)
        self.bn1 =  nn.BatchNorm2d(32)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#AlexCONV2(96, 256,k=5,s=1,p=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)#AlexPool2(k=3,s=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.relu2 = nn.ReLU()
        

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#AlexCONV3(256,384,k=3,s=1,p=1)
        self.bn2 =  nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)#AlexCONV5(384, 256, k=3, s=1,p=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool3(k=3,s=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(256*3*3, 1024)  #AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512) #AlexFC6(4096,4096)
        self.fc8 = nn.Linear(512, 10)  #AlexFC6(4096,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 256 * 3 * 3)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x

# MobileNet, test depthwise layer
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  1,  16, 2), 
            conv_dw( 16,  32, 1),
            #conv_dw(32, 64, 1),
            nn.AvgPool2d(2, 2),
            #nn.AdaptiveAvgPool2d((1,1))
            #nn.AdaptiveMaxPool2d((1,1))
            #nn.AvgPool2d(7, 7)
        )
        self.fc = nn.Linear(32*49, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 32*49)
        x = self.fc(x)
        return x
        
        
#Vggnet, test dropout, softmax layer.
class VggNet(nn.Module):
    def __init__(self):
        super(VggNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
        
        
transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),


])

transform1 = transforms.Compose([
                    transforms.ToTensor()
])


trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True,num_workers=0)

testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform1)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)

#net = MobileNet()
net = AlexNet()
#net = VggNet()

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
net.to(device)
print(net)
print("Start Training!")

num_epochs = 1

correct = 0
total = 0
for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 1024

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('[%d, %5d] loss:%.4f %.4f'%(epoch+1, (i+1)*100, loss.item(), 100*correct/total))

print("Finished Traning")

with torch.no_grad():
    
    correct_torch = 0
    total_torch = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = net(images)
        
        
        #predicted_keras = np.argmax(keras_output.data, 1)
        torch_output = output
       
        _, predicted_torch = torch.max(torch_output.data, 1)
       
        total_torch += labels.size(0)
        correct_torch += (predicted_torch == labels).sum().item()
        
    print('Accuracy of the torch network on the 10000 test images:{}%'.format(100 * correct_torch / total_torch))

model_name = 'MNIST_test'

#save the net parameters   
print("path_wd", path_wd)
torch.save(net.state_dict() ,os.path.join(path_wd,model_name +'.pth'))

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['input'] = {
    'model_lib': "torch"
}
config['restrictions'] = {
    'model_libs' : {'keras', 'lasagne', 'caffe', 'torch'} 
}   
config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': True              # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 300,             # How many test samples to run.
    'batch_size': 50,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['output'] = {
    'plot_vars': {                  # Various plots (slows down simulation).
        'spiketrains',              # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)
