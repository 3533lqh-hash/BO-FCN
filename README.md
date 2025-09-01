# BO-FCN
The combination of Bayesian optimization algorithm and semantic segmentation model.
What is this repository for?
BO-FCN is a software for paper 《Bayesian-optimized semantic segmentation model and its application in the identification of organic microcomponents》.

Who do I talk to?
Liu Qianghao；

College of earth Sciences, Jilin University, Jilin, Changchun 130061, China;
Key Laboratory of Oil Shale and Coexistent Energy Minerals of Jilin Province, Jilin, Changchun 130061, China E-mail: 335354068@qq.com;
Usage
Copy the created VOC format dataset and the training/testing partition files to the dataset folder.
Use the train.py file to train the model weights and generate the weight files.
Utilize the test.py file to set the basic parameters for training, enabling the optimization algorithm module to test the trained model using the prepared test dataset and output the test results.
Use the eval.py file to validate the validation dataset and output the validation data and images.
code introduction

Module Introduction:

    1. core: This directory contains the code for the semantic segmentation model as well as the skeleton network model.
    
    2. datasets: This folder is designated for storing the original dataset in VOC format.
    
    3. runs: This directory is used to store training log files and the predicted result images after testing and validation.
    
    4. scripts:
        (1) **checkpoints**: This folder is for storing the trained weight files.
        
        (2) **train.py**: This script is used to train the model using the training dataset and generate weight files.
        
        (3) **test.py**: This script tests the model using the test dataset and outputs the test results.
        
        (4) **eval.py**: This script validates the model using the validation dataset and outputs the validation data and images.
