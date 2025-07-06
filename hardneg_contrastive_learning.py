import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parallel import DataParallel
from data_utils import TransformedDataset, train_val_split, HardNegContrastiveDataset, MultiTransform, get_misclassified_images
from model import CNN
from contrastive_loss import HardNegContrastiveLoss
import torch.optim as optim
import wandb
from train_test_utils import hcl_train, cl_epoch_log

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cl_model', type=str, required=True)
    parser.add_argument('--sl_model', type=str, required=True)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num_positives', type=int, default=2)
    parser.add_argument('--num_hard_negatives', type=int, default=8)
    parser.add_argument('--num_randoms', type=int, default=54)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--contrast_mode', type=str, default='scl')
    parser.add_argument('--batch_size0', type=int, default=1024)
    parser.add_argument('--batch_size1', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--project', type=str, default='CIFAR-10-Classification-HardNeg-Contrastive-Learning')
    return parser.parse_args()


def main():
    args = arg_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(mean=-mean/std, std=1/std)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    pil_transform = transforms.Compose([
        denormalize,
        transforms.ToPILImage(),
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root='./data/train', transform=None)
    train_dataset, _ = train_val_split(dataset, val_size=0.2, seed=42)
    csf_train_set = TransformedDataset(dataset=train_dataset, transform=tensor_transform)
    csf_loader = DataLoader(csf_train_set, batch_size=args.batch_size0, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    csf_model = CNN(out_dim=10).to(device)
    csf_model = DataParallel(csf_model)
    csf_model.load_state_dict(torch.load(f'./models/{args.sl_model}.pth', map_location=device))

    misclassified_dict = get_misclassified_images(csf_model, csf_loader, pil_transform)

    hardneg_train_set = HardNegContrastiveDataset(dataset=train_dataset, 
                                                pos_transform=MultiTransform(transform=train_transform, num_transforms=args.num_positives), 
                                                hardneg_transform=train_transform, 
                                                hardneg_dict=misclassified_dict,
                                                num_hardnegs=args.num_hard_negatives,
                                                num_randoms=args.num_randoms,
                                                random_transform=train_transform)
    
    hardneg_loader = DataLoader(hardneg_train_set, batch_size=args.batch_size1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = CNN(out_dim=args.projection_dim).to(device)
    model = DataParallel(model)
    model.load_state_dict(torch.load(f'./models/{args.cl_model}.pth', map_location=device))
    model_dir = f'./models/hcl_{args.cl_model}.pth'

    criterion = HardNegContrastiveLoss(contrast_mode=args.contrast_mode, temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.login()
    wandb.init(project=args.project)
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    for epoch in range(1, args.epochs + 1):
        train_loss = hcl_train(epoch, hardneg_loader, device, model, criterion, optimizer, scheduler, log_freq=args.log_freq)
        cl_epoch_log(epoch, args.epochs, train_loss)

    torch.save(model.state_dict(), model_dir)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()