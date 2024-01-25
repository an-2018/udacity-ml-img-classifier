# train.py
import torch
import argparse
from utility import load_data, build_model, train_model, save_checkpoint
from torch import nn
from torch import optim

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    parser.add_argument('data_dir', help='dataset')
    parser.add_argument('--save_dir', dest='save_dir', default='./models', help='Directory to save checkpoints')
    parser.add_argument('--arch', dest='arch', default='vgg', help='Choose architecture (vgg16, densenet)')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', dest='epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training')

    args = parser.parse_args()

    train_loader, test_loader, valid_loader, train_data = load_data(args.data_dir)
    
    model = build_model(args.arch, args.hidden_units)
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)

    save_checkpoint(model, args.save_dir,  args.epochs, optimizer, train_data, args.arch)

if __name__ == "__main__":
    main()
