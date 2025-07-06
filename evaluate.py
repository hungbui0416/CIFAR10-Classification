import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from train_test_utils import test
from model import CNN
from torch.nn.parallel import DataParallel
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    return parser.parse_args()

def main():
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_set = datasets.ImageFolder(root='./data/test', transform=transform)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    classes = test_set.classes

    model = CNN(out_dim=10).to(device)
    model = DataParallel(model)
    model.load_state_dict(torch.load(f'./models/{args.model}.pth', map_location=device))
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



