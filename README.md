# Chest_Xray_normal_pneumonia_classification

The purpose of this repository is to detetect pneumonia(bacterial/viral) from X-Ray image using classification methods.
This repository has developed for Machine Learning course in my study.

We have created the following models:
1) Convolutional Neural Network (CNN)
2) Backpropagation algorithm
3) K-Nearest neighbor (KNN)
4) Recurrent Neural Networks (RNN)
5) AdaBoost
The main idea was to understand more how the models work, not to have high accuracy.
The models have been developed using Jupyter Notebook.

#### DataBase
Our Database has been taken from:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

This database has 5000+ images(JPEG), which fall into three categories:
* Train 
* Test
* Validation 

Every category has two types of images:
* Normal - depicts clear lungs
* Pneumonia - depicts pneumonia from viral pathogens or bacterial pathogens




Images example:


The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.



## Results:

#### CNN:


#### Backpropagation:


#### KNN:


#### RNN:


#### AdaBoost:


## Challenges:
The biggest challenge of this project is the imbalance of the dataset.
* The number of X-Ray images for NORMAL and PNEUMONIA cases is not 50%/50% in training and test datasets.
* The class ratio is not consistent across different datasets. The NORMAL/PNEUMONIA ratio was around 1:3 in training dataset so we made it 1:1 to be balanced, also 1:1 ratio in validation dataset, and around 1:1.67 in test dataset.
* The dataset is relatively small, thus may lead to overfitting and low prediction accuracy on test dataset.
* We tried to fit the model into 5 different algorithms, The first one is CNN (Convolution neural network), that can handle the images very well because there is a kernel that moves around the image and shares it's weight, and that’s why it was the best.
* The second is back propagation, fully connected neural network that can improve the weight by errors. (it reduces/increases the weights by the gradient of the error) so it can learn the data very well too. 
* The third was AdaBoost classifier, in this method we tried to work again on the data so the algorithm can handle it better (1, -1), (there is no neural network in this algorithm).
* The fourth is KNN cluster (K- Nearest neighbor), it takes an image and tries to find the closest 3 neighbors and clusters it to the closest label. 
* The last was RNN (recurrent neural network), although in this algorithm we tried to do our best but couldn’t handle the data .
* To find the best learning rate that can give the best results, an in some values we got a good accuracy on the training data but it wasn’t close in the test(overfitting) so we tried to  find the best value.
* To find the best normalized size to the image (150, 150) in a way that we don’t lose any features that are important to classification.
* To get the number of hidden layers in back Propagation that can learn the data without overfitting.
* Sometimes we got a great accuracy  


