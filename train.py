from model_loader import load_model, to_cuda_when_available
from image_loader import load_image

from torchvision import transforms, datasets, models
import torch
from torch import nn, optim
import numpy as np
import itertools
import math

import json
import time

import argparse

def train(data_directory, save_dir, is_gpu_available, model_architecture, learning_rate, hidden_units, epochs):
    train_dir = data_directory + '/train'
    validation_dir = data_directory + '/valid'
    # Define your transforms for the training and validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = data_transforms = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'validation': datasets.ImageFolder(validation_dir, transform=data_transforms['validation']),
    }

    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers = 6),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True, num_workers=2),
    }
    
    
    if (model_architecture == 'densenet121'):
        model = models.densenet121(pretrained = True)
        prevent_model_from_training(model)
        model.classifier = build_classifier(1024, hidden_units)
    if (model_architecture == 'vgg13'):
        model = models.vgg13(pretrained = True)
        prevent_model_from_training(model)
        model.classifier = build_classifier(25088, hidden_units)
    else:
        raise Exception(f"model {model_architecture} is not supported")
        
    [model] = to_cuda_when_available([model], is_gpu_available)
    [criterion] = to_cuda_when_available([nn.NLLLoss()], is_gpu_available)
    optimizer = optim.SGD(model.classifier.parameters(), lr = learning_rate, momentum=0.5)
    
    training_chunk_size = 5
    for epoch in range(epochs):
        # In each chunk we will be performing the training and evaluation against the validation set
        training_chunks = chunk_every(dataloaders['train'], training_chunk_size)

        for (training_index, training_chunk) in enumerate(training_chunks):
            # Perform training
            training_loss = 0
            model.train()
            for images, labels in training_chunk:
                [images, labels] = to_cuda_when_available([images, labels], is_gpu_available)
                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

            # Perform evaluation
            else:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for evaluation_images, evaluation_labels in dataloaders['validation']:
                        [evaluation_images, evaluation_labels] = to_cuda_when_available([evaluation_images, evaluation_labels], is_gpu_available)
                        output = model.forward(evaluation_images)
                        loss = criterion(output, evaluation_labels)

                        validation_loss += loss.item()

                        probabilities = torch.exp(output)
                        top_probabilities = torch.argmax(probabilities, 1)
                        accuracy += (top_probabilities == evaluation_labels).sum().item() / evaluation_labels.size(0)
                    print(f"Epoch {epoch + 1} / {epochs}")
                    print(f"Step {training_index + 1} / {math.ceil(len(dataloaders['train']) / training_chunk_size)} - training_loss:", round(training_loss / (len(dataloaders['train']) / training_chunk_size), 4))
                    print(f"Step {training_index + 1} / {math.ceil(len(dataloaders['train']) / training_chunk_size)} - validation_loss:", round(validation_loss / len(dataloaders['validation']), 4))
                    print(f"Step {training_index + 1} / {math.ceil(len(dataloaders['train']) / training_chunk_size)} - validation_accuracy:", round(accuracy / len(dataloaders['validation']), 4))

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'model': model,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx
    }

    torch.save(checkpoint, save_dir + str(round(time.time()))+ '_checkpoint.pth')

def build_classifier(in_features, hidden_units):
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim = 1),
    )
def prevent_model_from_training(model):
    for param in model.parameters():
        param.requires_grad = False

def chunk_every(iterable, chunk_size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk
    
def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', action='store',
                    help='directory containing training and validation data')

    parser.add_argument('--save_dir', action='store',
                    default='saved_checkpoints/',
                    type=str,
                    dest='save_dir',
                    help='Directory to save checkpoints, defaults to saved_checkpoints/')

    parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='is_gpu_available',
                    help='Should use GPU when training. Setting this option will make the model train ~10x faster')

    parser.add_argument('--arch', action='store',
                    default='densenet121',
                    dest='model_architecture',
                    type=str,
                    help='Model architecture, defaults to densenet121')
    
    parser.add_argument('--learning_rate ', action='store',
                    default=0.03,
                    type=float,
                    dest='learning_rate',
                    help='Learning rate, the model will generaly converge better for learning rates between 0.01 and 0.1')
    
    parser.add_argument('--hidden_units ', action='store',
                    default=256,
                    type=int,
                    dest='hidden_units',
                    help='number of hidden units')
    
    parser.add_argument('--epochs ', action='store',
                    default=4,
                    type=int,
                    dest='epochs',
                    help='number of epochs to train, 4 epochs should be enough to achieve ~80% accurary at the default learning_rate')
    
    results = parser.parse_args()
    return results.data_directory, results.save_dir, results.is_gpu_available, results.model_architecture, results.learning_rate, results.hidden_units, results.epochs

if __name__ == '__main__':
    data_directory, save_dir, is_gpu_available, model_architecture, learning_rate, hidden_units, epochs = get_parameters()
    train(data_directory, save_dir, is_gpu_available, model_architecture, learning_rate, hidden_units, epochs)