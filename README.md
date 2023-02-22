# AI Image Classifier Application
> I built an AI image classifier application to recognize different species of flowers using Pytorch, then convert it into a command line application as a requirement for Udacity's AI Programming with Python Nanodegree program.
> 

## Table of Contents
- [AI Image Classifier Application](#ai-image-classifier-application)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Technologies Used](#technologies-used)
  - [Features](#features)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Project Status](#project-status)
  - [Room for Improvement](#room-for-improvement)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)
  - [License](#license)


## General Information
- This AI application was trained on a dataset of 102 flower categories gotten from ImageNet. Using Transfer Learning i built a model which uses a deep learning model trained on hundreds of thousands of images as part of the overall application architecture.
- It identifies name of Flowers, going forward AI algorithms will be incorporated into more and more everday applications. This model can be integrated into a phone app that tells you the name of the flower your camera is looking at.


## Technologies Used
- Pytorch
- Jupyter Notebook
- Google Colab


## Features
 - Command Line application
 - Ready trained model


## Setup
- It is required that Python is already installed.
- Also note training data is not included in this repo
- Required dependencies are located in a file requirements.txt
        `pip install -r requirements.txt`


## Usage
- Clone this repo or download as zip file
- Open a commandline prompt, navigate to the folder directory
        `cd C:/Users/emmas/Desktop/emmanuel_udacity`
- Train the model, replace data_directory with flower dataset(Prints out training loss, validation loss and validation accuracy as it trains)
        `python train.py data_directory`
- Some options include:
    - Set directory to save checkpoints:
            ` python train.py data_dir --save_dir save_directory`
    - Set hyperparameters:
            `python train.py data_dir --learning_rate 0.01 – hidden_units 512 – epochs 20`
    - Use GPU for training:
            `python train.py data_dir --gpu`
    - Choose architecture:
            `python train.py data_dir --arch `
- Predict a flower name from a single image path, this returns the flower name and class probability:
        `python predict.py /path/to/image checkpoint`
- Options include:
    - Return top K most likely classes:
            ` python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real names:
            `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference:
            `python predict.py input checkpoint --gpu`


## Project Status
Project is: _complete_

## Room for Improvement
Room for improvement:
- This model can be further integrated into a Phone App, that uses the camera to identify and display name of flowers.


## Acknowledgements
- This project was based on a requirement for Udacity's AI Programming with Python Nanodegree program.
- Many thanks to Udacity and AWS for the opportunity.


## Contact
Created by [@Emmanuel-Samuel] - feel free to contact me!


## License
This project is available under the Udacity License.