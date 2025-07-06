from PIL import Image
import argparse
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torchvision import transforms
from model import CNN
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sl_hcl_scl',
                      help='Name of the model')
    parser.add_argument('--image_path', type=str, default='./data/test/dog/0001.png',
                      help='Path to the input image')
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    try:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found at {args.image_path}")
        
        image = Image.open(args.image_path)
        image_tensor = transform(image)
        
        model_path = f'./models/{args.model}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = CNN(out_dim=10).to(device)
        model = DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

        predicted_class = classes[predicted.item()]
        print(f"\nPredicted class: {predicted_class}")
        print("\nClass probabilities:")
        for i, class_name in enumerate(classes):
            probability = probabilities[0][i].item() * 100
            print(f"{class_name:10s}: {probability:.2f}%")

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()