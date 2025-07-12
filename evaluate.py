import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import argparse
import os

from networks import CNN, ResNet18

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='cnn', help='name of network')
    parser.add_argument('--model', type=str, required=True, help='path to the trained model')
    
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    return parser.parse_args()

def main():
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_set = datasets.ImageFolder(root='./data/test', transform=transform)
    loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    classes = test_set.classes

    if args.network == 'cnn':
        model = CNN(out_dim=len(classes)).to(device)
    elif args.network == 'resnet18':
        model = ResNet18(out_dim=len(classes)).to(device)
    else:
        raise ValueError(f"Unknown network: {args.network}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_true, y_pred = test(loader, device, model, criterion)
    print(f"\ntest_loss {test_loss:.4f} test acc {test_acc:.2f}%")

    print(f"\n{classification_report(y_true, y_pred)}")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()



