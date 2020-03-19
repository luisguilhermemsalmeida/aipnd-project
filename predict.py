from model_loader import load_model, to_cuda_when_available
from image_loader import load_image

import torch
import numpy as np

import json

import argparse

def predict(image_path, checkpoint_path, topk, is_gpu_available):
    image = load_image(image_path)
    model = load_model(checkpoint_path)
    [model] = to_cuda_when_available([model], is_gpu_available)
    
    with torch.no_grad():
        [image] = to_cuda_when_available([image.unsqueeze(0)], is_gpu_available)
        model.eval()
        output = model.forward(image)
        probabilities = torch.exp(output)

        top_k_probabilities, top_k_index = torch.topk(probabilities, topk)

        index_to_class = {value: key for key, value in model.class_to_idx.items()}

        top_k_classes = list(map(lambda index: index_to_class[index], np.array(top_k_index.cpu())[0]))
    return top_k_probabilities, top_k_classes

def pretty_print_prediction(top_k_probabilities, top_k_classes, category_name_mapping_file_path):
    with open(category_name_mapping_file_path, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = list(map(lambda class_number: cat_to_name[class_number] + " (" + class_number + ")", top_k_classes))
    
    print("Prediction Result")
    print("#################")
    for i in range(len(top_k_classes)):
        print(f"{class_names[i]}: {round(top_k_probabilities.cpu().numpy()[0][i] * 100, 1)}%")
    print("#################")
def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', action='store',
                    help='Image path to the flower image to be predicted')

    parser.add_argument('checkpoint_file_path', action='store',
                    help='Neural Network Checkpoint file path to be loaded for prediction')

    parser.add_argument('--top_k', action='store',
                    default=1,
                    type=int,
                    dest='top_k',
                    help='Top K most likely classes, defaults to 1')
    parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='is_gpu_available',
                    help='Should use GPU for inference')
    parser.add_argument('--category_name', action='store',
                    default='cat_to_name.json',
                    type=str,
                    dest='category_name_mapping_file_path',
                    help='File path containing mapping of categories to real names')
    
    results = parser.parse_args()
    return results.image_path, results.checkpoint_file_path, results.top_k, results.is_gpu_available, results.category_name_mapping_file_path


image_path, checkpoint_path, topk, is_gpu_available, category_name_mapping_file_path = get_parameters()

top_k_probabilities, top_k_classes = predict(image_path, checkpoint_path, topk, is_gpu_available)

pretty_print_prediction(top_k_probabilities, top_k_classes, category_name_mapping_file_path)