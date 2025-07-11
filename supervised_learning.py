import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from data_utils import TransformedDataset, train_val_split
from networks import CNN, ResNet18
from train_test_utils import EarlyStopper, train, validate, epoch_log

import argparse
import os
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')

    parser.add_argument('--network', type=str, default='cnn', help='name of network')
    parser.add_argument('--model_path', type=str, default='', help='path of pretrained model')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension')
    
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--cosine', action='store_true', help='use cosine annealing scheduler')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--early_stop', action='store_true', help='use early stopping')
    parser.add_argument('--patience', type=int, default=3, help='patience')
    parser.add_argument('--min_delta', type=float, default=0.1, help='minimum delta')

    parser.add_argument('--log_freq', type=int, default=10, help='log frequency')
    return parser.parse_args()


def main():
    # parse args
    args = parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        val_transform
    ])

    # train dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    # train/val split
    train_dataset, val_dataset = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, train_transform)
    val_set = TransformedDataset(val_dataset, val_transform)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model and optimizer
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    model_dir = f'./ckpts/sl_{args.network}.pth'

    if args.network == 'cnn':
        if args.model_path:
            model = CNN(num_classes=args.projection_dim).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(512*4*4, 10)
            optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            model = CNN(num_classes=10).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model = DataParallel(model)
    elif args.network == 'resnet18':
        if args.model_path:
            model = ResNet18(num_classes=args.projection_dim).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(512, 10)
            optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            model = ResNet18(num_classes=10).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model = DataParallel(model)
    elif args.network not in ['cnn', 'resnet18']:
        raise ValueError(f'Invalid network: {args.network}')
    elif not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model path does not exist: {args.model_path}')

    # criterion, scheduler and early_stopper
    criterion = nn.CrossEntropyLoss()
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    if args.early_stop:
        early_stopper = EarlyStopper(model, model_dir, patience=args.patience, min_delta=args.min_delta)
    else:
        early_stopper = None

    # wandb
    wandb.login()
    wandb.init(project='CIFAR-10-Classification-Supervised-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    # train
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch, train_loader, device, model, criterion, optimizer, scheduler, args.log_freq)
        val_loss, val_acc = validate(epoch, val_loader, device, model, criterion)
        epoch_log(epoch, args.epochs, train_loss, train_acc, val_loss, val_acc)

        if early_stopper and early_stopper.early_stop(val_loss):
            print('Early stop at epoch', epoch)
            break

    # log model to wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()