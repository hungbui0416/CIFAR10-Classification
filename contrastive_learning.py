import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from data_utils import MultiTransform, train_val_split, TransformedDataset
from networks import CNN, ResNet18
from contrastive_loss import ContrastiveLoss
from train_test_utils import cl_train, cl_epoch_log

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--views', type=int, default=2, help='number of views per image')
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')

    parser.add_argument('--network', type=str, default='cnn', help='name of network')
    parser.add_argument('--projection', type=int, default=128, help='projection dimension')

    parser.add_argument('--mode', type=str, default='scl', help='constrastive learning mode')
    parser.add_argument('--temperature', type=int, default=0.1, help='temperature for contrastive loss')
    
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing scheduler')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')

    parser.add_argument('--log_freq', type=int, default=10, help='log frequency')
    return parser.parse_args()

def main():
    args = arg_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    train_dataset, _ = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, MultiTransform(transform, num_transforms=args.views))

    loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    model_dir = f'./ckpts/cl_{args.network}.pth'

    if args.network == 'cnn':
        model = CNN(num_classes=args.projection).to(device)
    elif args.network == 'resnet18':
        model = ResNet18(num_classes=args.projection).to(device)
    else:
        raise ValueError(f"Unknown network: {args.network}")
    model = nn.DataParallel(model)
    
    criterion = ContrastiveLoss(mode=args.mode, temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    wandb.login()
    wandb.init(project='CIFAR-10-Classification-Supervised-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    for epoch in range(1, args.epochs + 1):
        train_loss = cl_train(epoch, loader, device, model, criterion, optimizer, scheduler, log_freq=args.log_freq)
        cl_epoch_log(epoch, args.epochs, train_loss)

    torch.save(model.module.state_dict(), model_dir)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()