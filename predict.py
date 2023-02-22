#                                                                             
# PROGRAMMER: EMMANUEL MAYOWA SAMUEL
# DATE CREATED: 21/02/2023
# REVISED DATE: 
# PURPOSE:  Predict the flower, from a single image path directory
# Prints out flower name, and class probability
# 
##




import argparse
import functions_model
import data_loader
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description="Predicting name of the  flower")
    parser.add_argument('image_path', action="store",
                        help='Image Path',
                        default='flower_data/train/67/image_07080.jpg')

    parser.add_argument('checkpoint', action='store',
                        help='Checkpoint to use',
                        default='./chkp.pth')

    parser.add_argument('--top_k', action='store', help='Top 5 probability results displayed',
                        default='5', dest='top_k', type=int)

    parser.add_argument('--category_names', action='store',
                        help='The directory of label mapping file - .json',
                        default='cat_to_name.json', dest='category_names')


    parser.add_argument('--gpu', action='store_true', help='Execute the training on gpu',
                        dest='gpu')
    
    
    args = parser.parse_args()
    
    # Checks if GPU is available for use
    if torch.cuda.is_available() and args.gpu :
        gpu = True
    else:
        gpu = False
        
    top_p, top_class, class_to_idx = model_functions.predict(args.image_path, args.checkpoint, args.top_k, args.gpu)
    
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
        
    reversed_class_to_idx = {class_to_idx[cl]: cl for cl in class_to_idx}
    
    mapped_classes = []
    names = []

    for label in top_class:
        mapped_classes.append(reversed_class_to_idx[label])

    for c in mapped_classes:
        names.append(cat_to_name[str(c)])

    for i in range(0, len(top_p)):
        print('Flower : {}  Probabliities {}'.format(names[i], top_p[i]))

if __name__ == "__main__":
        main()
    
    
    
    
    
