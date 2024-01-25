import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch import optim

def load_data(data_dir):
    # Load data and transform
    # (train_loader, valid_loader, test_loader) = ...
    print("Loading data from {}...".format(data_dir))
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
    
    print("Loading data...")
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    return train_dataloader, test_dataloader, valid_dataloader, train_data

def build_model(arch, hidden_units):
    # Build model
    print("Building Model for {}...".format(arch))
    classifier = None

    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        freeze_parameters(model)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(4096, 102)),
            ('batch_norm', nn.BatchNorm1d(102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 500)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(500, 102)),
            ('batch_norm', nn.BatchNorm1d(102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        print("choose a model for transfer learning: vgg16 or densenet121")

    model.classifier = classifier
    return model

def freeze_parameters(model):
    # Freeze parameters
    print("Freezing parameters...")
    for param in model.parameters():
        param.requires_grad = False

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    # Train the model
    print("Training Model with: epocs: {}, device: {} ...".format(epochs, device))

    model.to(device)
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class==labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_loader))
            valid_losses.append(valid_loss/len(valid_loader))
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_loader):.3f}.. "
                  f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()



def save_checkpoint(model, save_dir, epochs, optimizer, train_data, arch):
    # Save checkpoint
    print("Saving checkpoint at {}...".format(save_dir))
    
    if arch == 'vgg':
        model.class_to_idx = train_data.class_to_idx

        checkpoint = {'input_size': 25088,
                        'output_size': 102,
                        'hidden_layers': [4096, 102],
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'class_to_idx': model.class_to_idx,
                        'epochs': epochs}

    elif arch == 'densenet':
        model.class_to_idx = train_data.class_to_idx

        checkpoint = {'input_size': 1024,
                        'output_size': 102,
                        'hidden_layers': [500, 102],
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'class_to_idx': model.class_to_idx,
                        'epochs': epochs}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')

def load_checkpoint(filepath, arch):
    # Load checkpoint
    print("Loading checkpoint from {}...".format(filepath))
    checkpoint = torch.load(filepath)

    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(4096, 102)),
            ('batch_norm', nn.BatchNorm1d(102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 500)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(500, 102)),
            ('batch_norm', nn.BatchNorm1d(102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    # Process input image
    print("Processing image at {}...".format(image_path))
    from PIL import Image 
    open_image = Image.open(image_path)
    preprocess_img = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img_tensor = preprocess_img(open_image)

    return img_tensor.numpy()

def predict(image_path, model, topk, device):
    # Make predictions
    print("Predicting image at {}...".format(image_path))
    model.to(device)
    
    model.eval()
    
    image = process_image(image_path)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Move to the appropriate device
    image = torch.from_numpy(image).to(device)
    
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        
    return top_p, top_class

