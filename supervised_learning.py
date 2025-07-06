import argparse
from pydantic import validate_call_decorator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_utils import TransformedDataset, train_val_split
from torch.utils.data import DataLoader
from model import CNN
from torch.nn.parallel import DataParallel
from train_test_utils import EarlyStopper, train, validate, epoch_log
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=0.1)
    parser.add_argument('--log-freq', type=int, default=10)
    parser.add_argument('--project', type=str, default='CIFAR-10-Classification-Supervised-Learning')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # train dataset
    dataset = datasets.ImageFolder(root='./data/train', transform=None)

    # train/val split
    train_dataset, val_dataset = train_val_split(dataset, val_size=0.2, seed=42)
    train_set = TransformedDataset(dataset=train_dataset, transform=train_transform)
    val_set = TransformedDataset(dataset=val_dataset, transform=val_transform)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    if args.model == '':
        model = CNN(out_dim=10).to(device)
        model = DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_dir = f'./models/sl.pth'
    else:
        cl_model = CNN(out_dim=args.projection_dim).to(device)
        cl_model = DataParallel(cl_model)
        cl_model.load_state_dict(torch.load(f'./models/{args.model}.pth'))
        model = CNN(out_dim=10).to(device)
        model = DataParallel(model)
        model.module.encoder = cl_model.module.encoder
        for param in model.module.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.module.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_dir = f'./models/sl_{args.model}.pth'
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopper = EarlyStopper(model, model_dir, patience=args.patience, min_delta=args.min_delta)

    # wandb
    wandb.login()
    wandb.init(project=args.project)
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    # train
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch, train_loader, device, model, criterion, optimizer, scheduler, log_freq=args.log_freq)
        val_loss, val_acc = validate(epoch, val_loader, device, model, criterion)
        epoch_log(epoch, args.epochs, train_loss, train_acc, val_loss, val_acc)

        if early_stopper.early_stop(val_loss):
            print('early stop at epoch', epoch)
            break

    # log model to wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()