# script to train ANN for MNIST
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from models.model import VGG5_ANN, VGG5_SNN_BNTT
from models.test import test_img
import datetime
from collections import namedtuple
from utils.options import args_parser


# Data: augmented MNIST: 3 x 32 x 32
train_datapath = '~/scratch60/FedMDA/digit5/mnist/train_images'
test_datapath = '~/scratch60/FedMDA/digit5/mnist/test_images'
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.ImageFolder(train_datapath, transform=transform)
test_dataset = datasets.ImageFolder(test_datapath, transform=transform)

num_samples = 10000
sampler = RandomSampler(train_dataset, num_samples=num_samples)
train_dl = DataLoader(train_dataset, sampler=sampler, batch_size=32, drop_last=True)
# test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

args = args_parser()
if args.snn:
    net = VGG5_SNN_BNTT().cuda()
else:
    net = VGG5_ANN().cuda()

print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # use default momentum, weight decay
loss_func = torch.nn.CrossEntropyLoss()

epochs = 1
eval_every = 1
test_accuracies = []

Args = namedtuple("Args", "gpu bs verbose")
test_args = Args(0, 32, False)


for epoch in range(epochs):
    print("Epoch: ", epoch)
    net.train()
    epoch_loss = []

    for batch_idx, (images, labels) in enumerate(train_dl):
        batch_loss = []
        images, labels = images.cuda(), labels.cuda()
        net.zero_grad()
        log_probs = net(images)

        loss = loss_func(log_probs, labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    epoch_loss.append(sum(batch_loss)/len(batch_loss))
    print("Epoch loss: ", epoch_loss[-1])

    if epoch % eval_every == 0:
        print("Testing")
        net.eval()
        acc_test, loss_test = test_img(net, test_dataset, test_args)
        print("Round {:d}, Testing accuracy: {:.2f}".format(epoch, acc_test))
        test_accuracies.append(acc_test)
    
if args.snn:
    dir_name = "SNN"
else:
    dir_name = "ANN"
save_model_path = os.path.join('/home/zy264/scratch60/FedMDA/MDA/pretrained_source', dir_name, str(datetime.datetime.now().date()) + '.pth')
print("Saved to: ", save_model_path)
torch.save(net.state_dict(), save_model_path)
