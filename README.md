# Chest_Xray_normal_pneumonia_classification

The purpose of this project is to detect pneumonia(bacterial/viral) from X-Ray image using classification methods. We developed this project in a Machine Learning course in our studying framework.

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

![Untitled](https://user-images.githubusercontent.com/44744877/182027394-79c15b11-736d-41dd-a6bf-60b5e52ed5c0.png)



Imaging examples:
![Untitled](https://user-images.githubusercontent.com/44744877/182027087-a3c5c618-7bfa-4d6d-9368-5419afe34699.png)


The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.



## Results:

#### CNN:
![Untitled](https://user-images.githubusercontent.com/44744877/182027124-0a3f0e9a-4d31-49d9-b005-c0c140426353.png)


#### Backpropagation:
![Untitled](https://user-images.githubusercontent.com/44744877/182027162-675c15f4-fc90-4538-bb37-301f827b3911.png)


#### KNN:
![Untitled](https://user-images.githubusercontent.com/44744877/182027128-a4073eb0-5417-41c4-88e8-7354adfab28c.png)


#### RNN:
![Untitled](https://user-images.githubusercontent.com/44744877/182027146-459ca54b-814f-41da-ae4c-dad571e00b9b.png)


#### AdaBoost:
![Untitled](https://user-images.githubusercontent.com/44744877/182027139-095aae54-a357-4ae3-b060-a429325eeee0.png)


## Main challenges we faced:
The biggest challenge of this project is the imbalance of the dataset.
* The number of X-Ray images for NORMAL and PNEUMONIA cases was not 50%/50% in training and test datasets.
* The class ratio is not consistent across different datasets. The NORMAL/PNEUMONIA ratio was around 1:3 in training dataset so we made it 1:1 to be balanced, also 1:1 ratio in validation dataset, and around 1:1.67 in test dataset.
* The dataset is relatively small, thus may lead to overfitting and low prediction accuracy on test dataset.
* We tried to fit the model into 5 different algorithms, The first one is CNN (Convolution neural network), that can handle the images very well because there is a kernel that moves around the image and shares it's weight. This algorithm gave the higher accuracy hence it was the best.
* The second is back propagation, fully connected neural network that can improve the weight by errors. (it reduces/increases the weights by the gradient of the error) so it can learn the data very well too. 
* The third was AdaBoost classifier, in this method we tried to work again on the data so the algorithm can handle it better (1, -1), (there is no neural network in this algorithm).
* The fourth is KNN cluster (K- Nearest neighbor), it takes an image and tries to find the closest 3 neighbors and clusters it to the closest label. 
* The last was RNN (recurrent neural network), although in this algorithm we tried to do our best but couldn’t handle the data .
* It was challenging to find the best learning rate that can give the best results, in some values we got a high accuracy rate on the training data but it wasn’t close in the test(overfitting) so we tried to find the best value.
* To find the best normalized size to the image (150, 150) in a way that we don’t lose any features that are important to classification.
* To get the number of hidden layers in back Propagation that can learn the data without overfitting.
  


