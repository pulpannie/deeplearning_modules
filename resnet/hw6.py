import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time


########################################
# You can define whatever classes if needed
########################################

class IdentityResNet(nn.Module):

    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,64, 3, stride=1, padding=1)
        self.stage1_1 = stage1_block()
        self.stage1_2 = stage1_block()
        self.stage2_1 = stage2_block1(64,128)
        self.stage2_2 = stage2_block2(128,128)
        self.stage3_1 = stage2_block1(128,256)
        self.stage3_2 = stage2_block2(256,256)
        self.stage4_1 = stage2_block1(256,512)
        self.stage4_2 = stage2_block2(512,512)
        self.fc = nn.Linear(512, 10)
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################

    ########################################
    # You can define whatever methods
    ########################################


    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        x = self.conv1(x)
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = F.avg_pool2d(x,4,stride=4)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x

class stage1_block(nn.Module):
    def __init__(self):
        super(stage1_block, self).__init__()
        self.BN = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64,64,3,stride=1,padding=1)


    def forward(self,x):
        a1 = x
        x = self.BN(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.BN(x)
        x = F.relu(x)
        x = self.conv(x)
        x += a1
        return x

class stage2_block1(nn.Module):
    def __init__(self, in_d, out_d):
        super(stage2_block1, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_d)
        self.BN2 = nn.BatchNorm2d(out_d)
        self.conv1 = nn.Conv2d(in_d,out_d,3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(out_d,out_d,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_d,out_d,1,stride=2)

    def forward(self,x):
        x = self.BN1(x)
        x = F.relu(x)
        a1 = self.conv3(x)
        x = self.conv1(x)
        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x += a1
        return x

class stage2_block2(nn.Module):
    def __init__(self, in_d, out_d):
        super(stage2_block2, self).__init__()
        self.BN = nn.BatchNorm2d(in_d)
        self.conv1 = nn.Conv2d(in_d,out_d,3,stride=1,padding=1)

    def forward(self,x):
        a1 = x
        x = self.BN(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.BN(x)
        x = F.relu(x)
        x = self.conv1(x)
        x += a1
        return x


########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 16

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
model = net.to(dev)

# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)

        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()

        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)

        # set loss
        loss = criterion(outputs, labels)

        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()

        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end - t_start, ' sec')
            t_start = t_end

print('Finished Training')

# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' % (classes[i]), ': ',
          100 * class_correct[i] / class_total[i], '%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct) / sum(class_total)) * 100, '%')


