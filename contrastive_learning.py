import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_utils import MultiTransform, train_val_split
from torch.utils.data import DataLoader
from model import CNN   
from contrastive_loss import ContrastiveLoss
from torch.nn.parallel import DataParallel
import wandb
from train_test_utils import cl_train, cl_epoch_log

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log-freq', type=int, default=10)
    parser.add_argument('--project', type=str, default='CIFAR-10-Classification-Contrastive-Learning')
    parser.add_argument('--contrast_mode', type=str, default='scl')
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--projection_dim', type=int, default=128)
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

    dataset = datasets.ImageFolder(root='./data/train', transform=MultiTransform(transform, args.num_views))
    train_set, _ = train_val_split(dataset, val_size=0.2, seed=42)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = CNN(out_dim=args.projection_dim).to(device)
    model = DataParallel(model)
    model_dir = f'./models/cl_{args.contrast_mode}.pth'

    criterion = ContrastiveLoss(contrast_mode=args.contrast_mode, temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.login()
    wandb.init(project=args.project)
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    for epoch in range(1, args.epochs + 1):
        train_loss = cl_train(epoch, loader, device, model, criterion, optimizer, scheduler, log_freq=args.log_freq)
        cl_epoch_log(epoch, args.epochs, train_loss)

    torch.save(model.state_dict(), model_dir)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()