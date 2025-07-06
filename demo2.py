import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_utils import get_misclassified_images, save_misclassified_images
import argparse
import os
from model import CNN
from torch.nn.parallel import DataParallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sl_hcl_scl',
                      help='Name of the model')
    parser.add_argument('--dataset', type=str, default='test',
                      help='Dataset to use (train or test)')
    return parser.parse_args()

def main():
    args = parse_args()

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

    try:
        # Load model
        model_path = f'./models/{args.model}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = CNN(out_dim=10)
        model = DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        if args.dataset == 'train':
            dataset = datasets.ImageFolder(root='./data/train', transform=tensor_transform)
        elif args.dataset == 'test':
            dataset = datasets.ImageFolder(root='./data/test', transform=tensor_transform)
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")

        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

        misclassified_dict = get_misclassified_images(model, loader, pil_transform)

        save_misclassified_images(misclassified_dict, save_dir=f'./misclassified/{args.model}_{args.dataset}')

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()