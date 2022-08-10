# script to train ANN for MNIST
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from models.model import VGG9_ANN, VGG9_SNN_BNTT
from models.test import test_img
import datetime
from collections import namedtuple
from utils.options import args_parser
import wandb


# Data: office31 amazon: 3 x 300 x 300, 2817 samples
datapath = '~/scratch60/FedMDA/office31/amazon'
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
dataset = datasets.ImageFolder(datapath, transform=transform)
print("{} samples with dimension {}".format(len(dataset), dataset[0][0].size()))

train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

print(len(train_dl.dataset))

args = args_parser()
if args.snn:
    net = VGG9_SNN_BNTT(img_size=64, num_cls=31, timesteps=10).cuda()
else:
    net = VGG9_ANN(num_cls=31).cuda()

# print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # use default momentum, weight decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
loss_func = torch.nn.CrossEntropyLoss()

epochs = 50
eval_every = 1
reduce_every = 20
test_accuracies = []

Args = namedtuple("Args", "gpu bs verbose")
test_args = Args(0, 32, False)

if args.snn:
    trial_name = "office31-amazon-pretrain-snn"
else:
    trial_name = "office31-amazon-pretrain-ann"

if args.wandb:
    wandb.init(project="FedSNN-MDA", name=trial_name, config={"snn":True if args.snn else False})

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
    if epoch % reduce_every == 0:
        scheduler.step()

    if epoch % eval_every == 0:
        print("Testing")
        net.eval()
        acc_test, loss_test = test_img(net, test_dataset, test_args)
        print("Round {:d}, Testing accuracy: {:.2f}".format(epoch, acc_test))
        if args.wandb:
            wandb.log({"loss": loss_test, "acc": acc_test, "Round": epoch+1})
        test_accuracies.append(acc_test)

if args.snn:
    dir_name = "SNN"
else:
    dir_name = "ANN"
save_model_path = os.path.join('/home/zy264/scratch60/FedMDA/MDA/pretrained_source', dir_name, "office31-" + str(datetime.datetime.now().date()) + '.pth')
print("Saved to: ", save_model_path)
torch.save(net.state_dict(), save_model_path)
