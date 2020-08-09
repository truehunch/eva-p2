import subprocess
import urllib
import cv2
import os
import glob
import datetime
import matplotlib.pyplot as plt
import plotly.express as px

import numpy as np
import pandas as pd
import json

import torch
import torchvision
from torchsummary import summary 
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
import albumentations.pytorch as AP

from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import classification_report

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau

from tqdm import tqdm

from models.model import Net
from dataset import FlightDataset

train_transform = A.Compose(
    [A.PadIfNeeded(min_height=224, min_width=224, p=1.0),
     A.RandomCrop(height=224, width=224, p=1.0),
     A.HorizontalFlip(p=0.5),
     A.Normalize(mean=[0.5276, 0.5793, 0.6078], std=[0.1930, 0.1871, 0.2064],),
     #A.CenterCrop(height=224, width=224, p=1),
     AP.ToTensor()], 
     #A.CenterCrop(height=224, width=224, p=1)],
)

test_transform = A.Compose(
    [A.PadIfNeeded(min_height=224, min_width=224, p=1.0),
     A.Normalize(mean=[0.5276, 0.5793, 0.6078], std=[0.1930, 0.1871, 0.2064],),
     A.CenterCrop(height=224, width=224, p=1),
     AP.ToTensor()],
     #A.HorizontalFlip(p=1)], 
     #A.CenterCrop(height=224, width=224, p=1)]
)

def collate_fn(batch):
    img = [sample['img'] for sample in batch]
    target = [sample['target'] for sample in batch]
    return {'img': img, 'target': target}

def load():
    dataset = FlightDataset(train_transform=train_transform, test_transform=test_transform)
    sampler = WeightedRandomSampler(weights=dataset.df['sample_weights'].values, num_samples=len(dataset.df), replacement=False)
    batch_size = 4
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    validation_split = 0.75
    train_count = int(validation_split * len(dataset))
    validation_count = int((1 - validation_split) * len(dataset))

    if use_cuda:
        dataloader_args = {'sampler': sampler,
                        'shuffle': False, 
                        'batch_size': batch_size, 
                        'num_workers': 4, 
                        'pin_memory': True, 
                        'collate_fn': collate_fn}
    else:
        dataloader_args = {'sampler': sampler,
                        'shuffle': False,
                        'batch_size': batch_size, 
                        'collate_fn': collate_fn}

    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_args, drop_last=True)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_count + 1, validation_count])

    train_sampler = WeightedRandomSampler(weights=dataset.df['sample_weights'].iloc[train_dataset.indices].values, 
                                          num_samples=len(train_dataset.indices), 
                                          replacement=False)

    if use_cuda:
        train_dataloader_args = {'sampler': train_sampler,
                                'shuffle': False, 
                                'batch_size': batch_size, 
                                'num_workers': 4, 
                                'pin_memory': True, 
                                'collate_fn': collate_fn,
                                'drop_last': True}
    else:
        train_dataloader_args = {'sampler': train_sampler,
                                'shuffle': False,
                                'batch_size': batch_size, 
                                'collate_fn': collate_fn,
                                'drop_last': True}

    if use_cuda:
        test_dataloader_args = {'shuffle': True, 
                                'batch_size': batch_size, 
                                'num_workers': 4, 
                                'pin_memory': True, 
                                'collate_fn': collate_fn,
                                'drop_last': True}
    else:
        test_dataloader_args = {'sampler': sampler,
                                'shuffle': True,
                                'batch_size': batch_size, 
                                'collate_fn': collate_fn,
                                'drop_last': True}

    train_dataloader = DataLoader(train_dataset, **train_dataloader_args)
    test_dataloader = DataLoader(test_dataset, **test_dataloader_args)
    
    return {'device': device,
            'dataset': dataset, 
            'data_loader': data_loader, 
            'train_dataloader': train_dataloader, 
            'test_dataloader': test_dataloader}
    
def train(model, dataset, device, train_loader, optimizer, epoch, logs, scheduler=None):
    model.train()
    dataset.set_train()
    train_loss = 0
    pbar = tqdm(train_loader)
    train_len = len(train_loader.dataset)
    n_train_batches = train_len / train_loader.batch_size

    train_loss, acc = 0, 0

    y_target = []
    y_pred = []

    log = {}
    log['epoch'] = epoch
    log['batch_loss'] = []

    #if len(logs['train']) > 0:
    #    w = np.array([1 - logs['train'][-1]['report'][str(i)]['f1-score'] for i  in range(4)]) * dataset.weights
    #else:
    w = np.array(dataset.weights)

    log['class_weights'] = w.tolist()

    print('\n')
    print(f"{'Weights:':20s}{w}")
    weights = torch.from_numpy(w).float()
    weights = weights.to(device)

    print('\n')
    for batch_idx, batch in enumerate(pbar):
        images = batch['img']
        target = batch['target']

        y_target.extend(target)

        images = torch.stack(images)
        target = torch.as_tensor(target).long()

        # Move data to cpu/gpu based on input
        images = images.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Loss computation
        batch_loss = F.cross_entropy(output, target, weight=weights, reduction='mean')
        train_loss += (batch_loss / n_train_batches).item()

        # Backward pass)
        batch_loss.backward()

        # Gradient descent
        optimizer.step()

        # Predictions
        output = F.log_softmax(output, dim=-1)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        y_pred.extend(pred.detach().cpu().tolist())
        acc += (pred.eq(target.view_as(pred)).float().mean().item() / n_train_batches)

        # Step scheduler if scheduler is present
        if scheduler:
            scheduler.step()

        # Logging - updating progress bar and summary writer
        pbar.set_description(desc= f'TRAIN : epoch={epoch} acc: {100 * acc:.2f} loss: {train_loss:.5f}')
        log['batch_loss'].append(batch_loss.item())

    print('\n')
    report = classification_report(y_target, y_pred, zero_division=0, output_dict=True)
    for item in report:
        print(f'{(dataset.mapper[int(item)] if item in [str(key) for key in dataset.mapper.keys()] else item):20s} : {report[item]}')
    
    log['train_loss'] = train_loss
    log['report'] = report

    logs['train'].append(log)
    return train_loss, acc

def test(model, dataset, device, test_loader, epoch, logs):

    model.eval()
    dataset.set_eval()
    pbar = tqdm(test_loader)
    test_len = len(test_loader.dataset)
    n_test_batches = test_len / test_loader.batch_size

    test_loss, acc = 0, 0

    y_target = []
    y_pred = []

    log = {}
    log['epoch'] = epoch
    log['batch_loss'] = []

    weights = torch.from_numpy(np.array(dataset.weights)).float()
    weights = weights.to(device)

    print('\n')
    for batch_idx, batch in enumerate(pbar):

        images = batch['img']
        target = batch['target']

        y_target.extend(target)
        
        images = torch.stack(images)
        target = torch.as_tensor(target).long()

        # Move data to cpu/gpu based on input
        images = images.to(device)
        target = target.to(device)

        # Forward pass
        # Forward pass
        output = model(images)

        # Loss computation
        batch_loss = F.cross_entropy(output, target, weight=weights, reduction='mean')
        test_loss += (batch_loss / n_test_batches).item()
        
        # Predictions
        output = F.log_softmax(output, dim=-1)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        y_pred.extend(pred.detach().cpu().tolist())
        acc += (pred.eq(target.view_as(pred)).float().mean().item() / n_test_batches)
        
        # Logging - updating progress bar and summary writer
        pbar.set_description(desc= f'TEST : epoch={epoch} acc: {100 * acc:.2f} loss: {test_loss:.5f}')
        log['batch_loss'].append(batch_loss.item())

    print('\n')
    report = classification_report(y_target, y_pred, zero_division=0, output_dict=True)
    for item in report:
        print(f'{(dataset.mapper[int(item)] if item in [str(key) for key in dataset.mapper.keys()] else item):20s} : {report[item]}')
    
    log['test_loss'] = test_loss
    log['report'] = report

    logs['test'].append(log)
    return test_loss, acc

def run(experiment_name):
    load_info = load()
    model = Net()
    model = model.to(load_info['device'])
    dataset = load_info['dataset']
    device = load_info['device']
    train_dataloader = load_info['train_dataloader']
    test_dataloader = load_info['test_dataloader']

    now = datetime.datetime.now()
    prefix = now.strftime('%m-%d-%y %H:%M:%S')

    log_file = f'log_{experiment_name}/log_{prefix}_{experiment_name}.json'

    model_dir_suffix = f'model_{experiment_name}'

    lr = 1e-2
    epochs = 45
    momentum = 0.7

    # optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=1500)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    best_test_accuracy = 0
    best_test_loss = float('inf')
    best_model_path = ''

    logs = {}
    logs['train'] = []
    logs['test'] = []

    for epoch in range(0, epochs):
        
        train_loss, train_accuracy = train(model, dataset, device, train_dataloader, optimizer, epoch, logs, scheduler=None)
        test_loss, test_accuracy = test(model, dataset, device, test_dataloader, epoch, logs)
        

        if test_accuracy > best_test_accuracy:
            best_model_path = f'{model_dir_suffix}/acc_{prefix}_{best_test_accuracy:2.2f}_epoch_{epoch}.pth'
            torch.save(model, best_model_path)
            
            best_test_accuracy = test_accuracy 
            best_test_loss = test_loss
        
        with open(log_file, 'w') as outfile:
            json.dump(logs, outfile)
        
        scheduler.step()