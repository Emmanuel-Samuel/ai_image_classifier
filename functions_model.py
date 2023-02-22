#                                                                             
# PROGRAMMER: EMMANUEL MAYOWA SAMUEL
# DATE CREATED: 21/02/2023
# REVISED DATE: 
# PURPOSE:  Stores all necessary functions required
# 
# 
##



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import json
import os
import PIL 
import matplotlib.pyplot as plt
import seaborn as sb


# Function that selects model architecture

def arch_selection(arch):
    
    selector = {
        'vgg13': models.vgg13(pretrained=True),
        'vgg16': models.vgg16(pretrained=True),
        'vgg19': models.vgg19(pretrained=True)
    }
     
        
    sel_arch = selector.get(arch, 0)
    if sel_arch == 0:
        print('Model is not recognised. Using default VGG16')
        return models.vgg16(pretrained=True)
    else:
        return sel_arch

# Function that takes input and train the model
    
def model_train(data_set, class_to_idx, hidden_units, learning_rate, epochs, arch, gpu, save_dir):
    
    model = arch_selection(arch)

    features = model.classifier[0].in_features
    
    classifier = nn.Sequential( nn.Linear(features, hidden_units),
                                            nn.Dropout(.20),
                                            nn.ReLU(),
                                            nn.Linear(hidden_units, 102),
                                            nn.LogSoftmax(dim=1))


    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.class_to_idx = class_to_idx
    
    if gpu == True:
        device = torch.device('cuda')
        
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    
# Defines the parameters for model training and evaluation   
    
    training_loss = 0
    test_loss = 0
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in data_set[0]:
            
            if gpu == True:
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

                
                
                
            with torch.no_grad():
                test_loss = 0
                accuracy = 0
                model.eval()
                    
                    
                for inputs, labels in data_set[1]:
                            
                    if gpu == True:
                        inputs = inputs.to(device)
                        labels = labels.to(device)                        
                        
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        

                print(
                    f"Epoch {epochs}.. "
                    f"Train loss: {running_loss/len(data_set[0]):.3f}.. "
                    f"Test loss: {test_loss/len(data_set[1]):.3f}.. "
                    f"Test accuracy: {accuracy/len(data_set[1]):.3f}")
            

# Functions that saves checkpoint to a file


    model.cpu()
    path_dir = os.path.join(save_dir, 'chkp.pth')
    
    torch.save({'class_to_idx': model.class_to_idx,
                'arch' : arch,
                'model_state_dict': model.state_dict(),
                'classifier' : classifier,
                'optimizer' : optimizer,
               'optimizer_dict': optimizer.state_dict()},
               working_dir)
    
    
    print("Successfull. The path of file{}".format(path_dir))
    
    
# Functions load the saved checkpoint


def load_model(checkpoint):
    saved = torch.load(checkpoint)
    arch = saved['arch']
    model = arch_selection(arch)


    features = model.classifier[0].in_features



    classifier = nn.Sequential( nn.Linear(features, 512),
                                            nn.Dropout(.20),
                                            nn.ReLU(),
                                            nn.Linear(512, 102),
                                            nn.LogSoftmax(dim=1))


    model.classifier = classifier
    model.load_state_dict(saved['model_state_dict'])
    model.class_to_idx = saved['class_to_idx']

    model.load_state_dict(saved['model_state_dict'])

    optimizer = saved['optimizer']
    optimizer.load_state_dict(saved['optimizer_dict'])

    return model, model.class_to_idx


# This funstions preprocess the image and converts into an object that can be used as input to the model
    
    
def process_image(image_path):
    
    from PIL import Image

    img = Image.open(image_path)

    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))


    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    img = np.array(img)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std

    img = img.transpose((2, 0, 1))

    return img


# This function predicts the class from an image file

def predict(image_path, save_dir, top_k, gpu):
    
    model, class_to_idx = load_model(save_dir)
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    if gpu == True:
        device = torch.device('cuda')
        model = model.to(device)
        image = image.to(device)
        
    logps = model(image)
    prob = torch.exp(logps)
    top_p, top_class = prob.topk(top_k, dim=1)
    
    classes = top_class[0].tolist()
    probs = top_p[0].tolist()
    return probs, classes, class_to_idx
    
    
    
    
    
    
    
    

                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
