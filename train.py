#                                                                             
# PROGRAMMER: EMMANUEL MAYOWA SAMUEL
# DATE CREATED: 21/02/2023
# REVISED DATE: 
# PURPOSE:  Train the model, replace data_directory with flower dataset
# (Prints out training loss, validation loss and validation accuracy as
# it trains)
##


import argparse
import torch
import dataloader
import functions_model
import torch
import os
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def main():
    
    parser = argparse.ArgumentParser("Trains and predict types of flowers")
    
    parser.add_argument('data_dir', help='The directory to file data',
                        metavar='DIR', default ='flower_data')
    
    parser.add_argument('--save_dir',help='The directory to save data', metavar='DIR',
                        default='./', dest='save_dir')

    parser.add_argument('--arch', action='store', help='The version of VGG neural network architecture',
                        default='vgg16', dest='arch')

    parser.add_argument('--hidden_units', action='store', type=int,
                        help='The Hidden units',
                        default=512, dest='hidden_units')

    parser.add_argument('--learning_rate', action='store', type=float,
                        help=' The Learning rate', default=.003,
                        dest='learning_rate')

    parser.add_argument('--epochs', action='store', type=int,
                        help='The number of epochs', dest='epochs',
                        default=2)

    parser.add_argument('--gpu', action='store', help='The model training on gpu',
                        dest='gpu', default=False)    
    
    
    args = parser.parse_args()
    

    
    
    
    if args.gpu == True:
        gpu = True
    else:
        gpu = False
        
    data_set, class_to_idx  = data_loader.load_data(args.data_directory)
    
    model_functions.model_train(data_set, class_to_idx, args.hidden_units, args.learning_rate, args.epochs, args.arch, gpu, args.save_dir)
    
    
if __name__ == "__main__":
        main()
