import os
from typing import Optional
from glob import glob
import numpy as np
import imageio
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import albumentations as A

from models import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="Fixed random seed for training")
parser.add_argument("--model", type=str, help="Type of model to use")
parser.add_argument("--tag", type=str, help="Tag for experiments")
parser.add_argument("--weights", type=str, help="Pretrained weights for training", default=None)
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),    
])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(),
    transforms.RandomResizedCrop(size=200, scale=(0.5, 1.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),    # ImageNet-pretrain transform
])

NORM_PARAM = 50

class UTKFaceDataset(torch.utils.data.Dataset):

    class2gender = {
        0: 'Male',
        1: 'Female'
    }

    class2race = {
        0: 'White',
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Others'
    }

    def __init__(self, root, split_root, partition, transform=None, const=False):
        """
        The dataset must have the following structure:
        
        <root>/<filename>.jpg, where <filename> = "[age]_[gender]_[race]_[date&time].jpg"
        <split_root>/<partition>_names.txt, where <partition> is given as a keyword argument to __init__ 
            and typically is one of: "train", "val", "holdout". 
            The file contains basenames separated by a newline.

        NB: landmarks reading is not supported yet.
        """
        super().__init__()

        self.root = root
        self.const = const
        self.transform = transform
        self.short_names = open(os.path.join(split_root, f"{partition}_names.txt")).readlines()
        self.short_names = [name.strip() for name in self.short_names if name]
        self.filenames = [os.path.join(root, name) for name in self.short_names]
        self.filenames = [full_name for full_name in self.filenames if os.path.exists(full_name)]
        print('filenames before extension:', len(self.filenames))
        
        self.samples = []

        print("Preloading files...")
        for fn in self.filenames:
            #fn = self.filenames[idx]
            img = np.array(imageio.imread(fn))
            img = img.astype(np.float32) / 255.0
            age = fn.split("/")[-1].split("_")[0]
            age = int(age)
                        
            self.samples.append((fn, img, age))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
#         fn = self.filenames[idx]
#         img = np.array(imageio.imread(fn))
#         img = img.astype(np.float32) / 255.0
#         age = fn.split("/")[-1].split("_")[0]
#         age = int(age)
        
        fn, img, age = self.samples[idx]
        
        if self.transform:
            img = self.transform(img)
            
        t = torch.ones([1, 200, 200])
        const = (random.random() < 0.9) or self.const
        if const:
            t *= -50
        else:
            t *= age
            
        img = torch.cat([img, t / NORM_PARAM])

        return img, {"age" : age, "name" : fn}

test_data = UTKFaceDataset(root="./UTKFace", split_root="./splits", partition="ages_test", transform=transform_test, const=True)
train_data = UTKFaceDataset(root="./UTKFace", split_root="./splits", partition="ages_train", transform=transform_train, const=False)
# ood_data = UTKFaceDataset(root="./UTKFace", split_root="./splits", partition="ages_ood", transform=transform_test)

import torch
import torchvision
import numpy as np
import random
import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 10
n_finetuning = 10
batch_size_train = 128 #128
batch_size_test = 128 #128
learning_rate = 1e-3
momentum = 0.5
log_interval = 100
class_num = 1 #20

device = "cuda"

random_seed = int(args.seed)
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

torch.manual_seed(random_seed)
random.seed(random_seed)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False, num_workers=64)

if args.model == "reg":
    print("Create regular model...")
    network = NoiseNet(class_num, init_ch=4).to(device)
elif args.model == "mcdp":
    print("Create MC-Dropout model...")
    network = NoiseNet(class_num, 0.1).to(device)
    
print(f"Model class: {type(network)}")
    
# if args.weights is not None:
#     network.load_state_dict(torch.load(args.weights))
    
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

min_age = 0 # 20
max_age = 80 # 60

def SigmaLoss(outputs, target):
    sigma = torch.nn.functional.elu(outputs[:, 1]) + 1.2
    loss = ((outputs[:, 0] - target) ** 2 / sigma ** 2 + torch.log(sigma)).mean()
    return loss

def year2bins(age):
    # bins = np.linspace(20, 60, class_num).astype("int")
    bins = np.linspace(min_age, max_age, class_num).astype("int")
    classes = np.digitize(age.cpu(), bins, right=True) - 1
    return torch.tensor(classes).to(age.device)

def train(epoch):
    i = 0
    correct = 0 
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        target = target["age"].to(device).long()
        transformed_ages = year2bins(target)

        data = data.to(device)
        output = network(data)
        loss = torch.nn.CrossEntropyLoss()(output, transformed_ages)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1][:, 0]
        c = (transformed_ages == pred).sum()
        correct += c

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc {:.3f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), c.item() / batch_size_train))
            train_losses.append(loss.cpu().item())
            
    network.eval()

def trainMAE(epoch):
    i = 0
    correct = 0 
    mean_loss = 0

    network.train()
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        target = target["age"].to(device).float()[:, None]
        data = data.to(device)
        output = network(data)
#         loss = SigmaLoss(output, target[:, 0])
        loss = torch.nn.MSELoss()(output[:, 0], target[:, 0])
        loss.backward()
        optimizer.step()
        mean_loss += loss.cpu().detach()
    
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} MLoss {:.3f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), mean_loss / (batch_idx + 1)))
            train_losses.append(loss.cpu().item())
    network.eval()

def age_regression(output, pow=1):
    prob = torch.nn.functional.softmax(output, dim=1)
    bins = np.linspace(min_age, max_age, class_num).astype("int")
    range_x = torch.tensor((bins[1:] + bins[:-1]) / 2).to(device)
    result = (range_x[None] ** pow * prob).sum(dim=1)
    # result = range_x[output.argmax(dim=1).cpu()]
    #return torch.tensor(result).to(device).float()
    return result.float()

def testMAE():
    loss_fn = lambda x, y: torch.abs(x - y).sum()
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader, total=len(test_loader)):
            data = data.to(device)
            output = network(data)
            target = target["age"].to(device).float()
            test_loss += torch.nn.L1Loss(reduction = 'sum')(output[:, 0], target).cpu()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print("TEST MAE: ", test_loss)
    return test_loss

def test2MAE():
    loss_fn = lambda x, y: torch.abs(x - y).sum()
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader, total=len(test_loader)):
            data = data.to(device)
            data[:, [-1]] = target["age"][:, None, None, None].float().tile([1, 1, 200, 200]).cuda() / NORM_PARAM
            pred = network(data.cuda())
            output = pred
            target = target["age"].to(device).float()
            test_loss += torch.nn.L1Loss(reduction = 'sum')(output[:, 0], target).cpu()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print("TEST 2 MAE: ", test_loss)
    return test_loss

def test():
    loss_fn = lambda x, y: torch.abs(x - y).sum()
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader, total=len(test_loader)):
            data = data.to(device)
            output = age_regression(network(data))
            target = target["age"].to(device)
            test_loss += loss_fn(output, target).cpu()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print("TEST MAE: ", test_loss)
    return test_loss


bestMAE=100
best_model = network.state_dict()
       
testMAE()
for epoch in range(1, n_epochs + 1):
    trainMAE(epoch)
    mae = testMAE()
    test2MAE()

    if bestMAE > mae:
        print("Update MAE: ", mae) 
        bestMAE = mae
        best_model = network.state_dict()

for g in optimizer.param_groups:
    g['lr'] = 1e-4

for epoch in range(1, n_finetuning + 1):
    trainMAE(epoch)
    mae = testMAE()
    test2MAE()

    if bestMAE > mae:
        print("Update MAE: ", mae) 
        bestMAE = mae
        best_model = network.state_dict()

s = random_seed
MODEL_NAME = f"{args.tag}_{args.model}_wo_unc_{s}.pth"
print(f"Saving model: {MODEL_NAME}...")
torch.save(best_model, "./weights/" + MODEL_NAME)