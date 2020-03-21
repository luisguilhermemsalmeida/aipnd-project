import torch
from torch import optim

def load_model(chekpoint_path):
    checkpoint = torch.load(chekpoint_path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.05, momentum=0.5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def to_cuda_when_available(items, is_gpu_available):
    device = torch.device("cuda" if is_gpu_available else "cpu")
    
    transformed_items = []
    for item in items:
        transformed_items.append(item.to(device))
    return transformed_items