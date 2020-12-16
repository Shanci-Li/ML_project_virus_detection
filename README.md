# ML_project_virus_detection
Author: Kang Fu, Shanci Li, Runke Zhou

# Introduction
This is the second project in CS-433 Machining Learning. In this project, we dealt a real-world challenge to accomplish the image segmentation of adenovirus particles in food vacuoles of eukaryotic organisms with machine learning methods offered by the environmental chemistry lab (LCE) of EPFL.
# Structure of Project
## Structure of data folder
You should keep the following data structure otherwise there will be some problems when running the following part. The convenient way is to just download the data from our Repo.
```
data
└───image_and_masks
    └───train_imgs
    |   └───train
    |   |   │  images1
    |   |   |  images2
    |   |   |  ...
    └───train_masks
    |   └───train
    |   |   │  mask1
    |   |   |  mask2
    |   |   |  ...
    └───val_imgs
    |   └───val
    |   |   │  image1
    |   |   |  image2
    |   |   |  ...
    └───val_imgs
    |   └───val
    |   |   │  mask1
    |   |   |  mask2
    |   |   |  ...
    └───test_imgs
    |   └───test
    |   |   │  image1
    |   |   |  image2
    |   |   |  ...
    └───pred
        | pred_mask1
        | pred_mask2
```
## Structure of label_images
- image_cut.py : cut the image into 512x512 and transformed into depth of 24 (the original image is 2048x2048 with the depth of 8)
- labelling.py : the script for labelling the image in the jupyter notebook
- labeling.ipynb : the jupyter notebook to provide the interactive labelling tool.<br/>
The last two files is referred from Ian Hunt-Isaak(https://youtu.be/aYb17GueVcU).
## Structure of training_model
- segmentation.py : create a class structure to implement the model training and predicting 
- run.py : a script that has been loaded with our final mode. you can run it to get the prediction masks of the images in the test_imgs/test. 
- training_example.iynb : a jupyter notebook to train the model
# Instructions
All the following setups are tested within Windows system.
## Step1: Cut and Label images
The size of original image provided 2048x2048 with the depth of 8. We have segmented it into size of 512x512 and transformed into depth of 24. This can be down using the image_cut.py. 
```bash
python image_segment.py <path of image> <path to store the output segmented image>
```
1. install the require the libraries for labelling.py and labelling.ipynb. You can find the relating libraries and versions in the requirement.txt.
When you finish installing the required libraries, you need to input the following code in the anaconda prompt to add the jupyter extension to ensure the labelling.ipynb can working properly. The execution of the following lines needs Node.js and npm. You should install before running.
```bash
conda install nodejs
pip install npm
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-sidecar jupyter-matplotlib
```
2. open the labelling.ipynb in the jupyter lab or jupyter notebook
3. run the code and the interactive label tool will show at the right side.
The input image size is 512*512 with a depth of 24. Regarding the size of the image, it can be changed in labelling.py
There is an video for labelling demo instruction, available at https://youtu.be/n9svkTR7jq4. This part of work is refered by Ian Hunt-Isaak(https://youtu.be/aYb17GueVcU).

## Step2: Train models
The method we adopt is transfer learning with U-net. U-net is a convolutional neural network that was initially designed to segment biomedical images. It is easy to implement with the help of segmentation_models library(https://github.com/qubvel/segmentation_models).
### Train your preferred model
There are a file called training_example.ipynb. This jupyter notebook will lead you to train the model by yourself. You can change the parameters offered by U-net functions in the segmentation_models library. You can also see the prediction mask of the training data as well as the training history data plot of loss and  intersection  over  union(IoU). Lastly, you can apply the model to predict the unseen data and see the results.
### Use our final model
If you do not want to train the model, you can use our final model in the folder of training model. Then run the run.py (check the data structure before running)
```bash
python run.py
```
After running this file, a prompt window will show the input image and the predicted masks. You can close that window to see the next one. Meanwhile, all the predicted maasked will be saved at the subfolder called pred in the data folder.
# Reference
https://towardsdatascience.com/how-we-built-an-easy-to-use-image-segmentation-tool-with-transfer-learning-546efb6ae98
