# Matlab-AI-Challenge


https://youtu.be/SI50UsL1Cco?si=XAWoM3Qlhaz6eU1r



#  Alzheimer's Disease Classification using ResNet-50 (MATLAB)
This project demonstrates how to classify Alzheimer's disease stages using brain MRI images and a transfer learning approach based on a pre-trained ResNet-50 model. It is implemented in MATLAB using Deep Learning Toolbox.
## Project Overview
- Input: Brain MRI images organized in class-labeled folders  
- Model: Pre-trained ResNet-50 (ImageNet)  
- Learning: Transfer Learning + Data Augmentation  
- Output: Multi-class classification (e.g., NonDemented, MildDemented, etc.)  
- Visualization: Prediction results + ROC Curves
---
##  Key Features
###  Data Preparation
- Uses `imageDatastore` for automatic label extraction from folder names
- Splits dataset into 70% training and 30% validation
- Converts grayscale to RGB images (`gray2rgb`)
###  Model Customization (Transfer Learning)
- Loads ResNet-50 and removes original classification layers (`fc1000`)
- Adds new fully connected, softmax, and classification layers
- Connects new layers using `layerGraph`
###  Data Augmentation
Enhances generalization performance using:
- Random X/Y reflection
- Random translation (±50 pixels)
- Random rotation (±30 degrees)
###  Training Setup
- Optimizer: Stochastic Gradient Descent with Momentum (SGDM)
- Mini-batch size: 8  
- Epochs: 10  
- Learning rate: 0.001  
- Includes real-time training plot
###  Evaluation & Visualization
- Displays random prediction results (3×3 subplot)
- Generates ROC curves with AUC for each class
---
## Dataset Structure (Example)
Dataset/
├── NonDemented/
│   ├── image1.jpg
│   └── ...
├── MildDemented/
│   ├── image2.jpg
│   └── ...
├── VeryMildDemented/
│   └── ...
