
<a name='0'></a>

# ConvNets Architectures - AlexNet
Most ConvNets architectures that we will cover were the results of the ![ImageNet Challenge](https://image-net.org/challenges/LSVRC/index.php) that happened from 2010 to 2017, but the most exciting things happened between 2012-2017 and that's exactly what we are going to dive deep into.

Below image give a high-level overview of the revolution of the CNNs architectures on ImageNet Challenge.
Most ConvNets architectures that we will cover were the results of the  ![ImageNet Challenge](https://image-net.org/images/logo.png) that happened from 2010 to 2017, but the most exciting things happened between 2012-2017 and that's exactly what we are going to dive deep into.

Below image give a high-level overview of the revolution of the CNNs architectures on ImageNet Challenge.




### What's in here: 

* [1. Deep Convolutional Neural Networks(AlexNet)](#1)
* [2. AlexNet Architecture](#2)
* [3. Reducing Overfitting](#3)
* [4. Learning Representations](#4)
* [5. AlexNet Implementation](#5)
* [6. Final Notes](#6)

<a name='1'></a>
## 1. Deep Convolutional Neural Networks(AlexNet)

[AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) is a Convolutional Neural Networks architecture that won the ImageNet Challenge of 2012 with top-error of `16.4%`. The current top-5 error rate on ImageNet was `25.8%`, so, AlexNet was pretty remarkable. 

AlexNet is one of the most influential paper in the field of deep learning and computer vision. As of 2021 on Google Scholar, AlexNet has been cited 99,354.

<a name='2'></a>

## 2. AlexNet Architecture
AlexNet architecture is made of 8 layers in total, five convolutional layers(3 of them followed by maxpooling layer), and 3 fully connected layers with a final layer made of 1000 units (ImageNet dataset had 1000 classes) and softmax activation function. 

AlexNet was very much similar to ![LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) that was introduced earlier before in 1990s except that it was deeper, and instead of using sigmoid activation functions in hidden layers like LeNet-5, it used ReLU.

Also, to reduce overfitting, AlexNet was the first paper to use dropout regularization technique. Up to day, dropout is one of the most effective regularization technique.
*Image: AlexNet architecture. The network is splitted across two GPUs*

As seen from the architecture:     

* The input image shape is (3,224,224) or (224,224,3)
* The first convolutional layer has 96(48+48 because it was split across two GPUs) filters, each filter with 11X11 size, and stride of 4. The output size(HXW) of feature maps for this layer is 54X54. 

* The first maxpooling layer following first convolutional layer has a pool size of 3 and stride 2. 

* The second convolutional layer has 256 filters, each filter having a size of 5X5.

* The third, fourth, and fifth convolutional layers are connected in series and they don't have maxpooling layers. Their respective number of filters are 384, 384, and 256. All filters with 3X3. 

* The second maxpooling layer has also pool size of 3 and strides of 2.

* The activation maps from previous maxpooling layer are flattned to be converted into 1D vector format which is a perfect format for the next fully connected layers. Flattening layer has no any learnable parameters.

* The two fully connected layers that follow flattening layers contain 4096 units each.  
* The last fully connected layer is 1000 units since the ImageNet dataset has 1000 classes and it has a softmax activation function.

AlexNet uses ReLU activation function in all hidden layers. This is because ReLU is so fast compared to sigmoid (sigmoid was being used a lot at the time). The issues with sigmoid is that it can cause the gradients to vanish early on in the network and the learning can die completely. ReLU overcomes that and it helps the model to train faster.

That's pretty much all about AlexNet.

<a name='3'></a>

## 3. Reducing Overfitting


In order to reduce overfitting, AlexNet used two main strong regularization techniques that are data augmentation and dropout. 

* The idea of data augmentation is to expand the training set by creating artificial images from the training images. The main two data augmentation techniques that were used in AlexNet are horizontal clipping and changing color intensity.

* Alex and his colleagues used dropout to make the network more robust. The idea of dropout is to randomly drop the neurons in hidden layers at a given probability which is 0.5 in AlexNet. All dropped-out neurons don't count during forward and backward propogation. Also, as the network is forced to learn without some neurons, it becomes more robust. Dropout can be seen as a computationally free ensembling technique since it creates a number of different networks with shared weights. AlexNet uses two dropout layers of 0.5 probability in the first two fully connected layers.

* <a name='4'></a>

## 4. Learning Representations
One of the main intringuing things about convolutional neural networks is that they don't need to operate on manual handcrafted features. They instead learn features or representations from the input image. 

The first convolutional layer of AlexNet learns low-level things like line, edges, colors, and textures. The higher layers learns high-level features such as nose, face, eyes, etc...

Below image(taken from AlexNet paper) depicts the features learned by the first layer in the AlexNet.

<a name='5'></a>

## 5. AlexNet Implementation
Below we will implement AlexNet. It's so simple with Pytorch Sequential API. 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Example instantiation
model = AlexNet(num_classes=1000)
```
<a name='6'></a>

## 6. Final Notes and Further Learning
AlexNet is one of the first CNNs architectures that influenced deep learning and computer vision. AlexNet proved that CNNs can achieve excellent results on range of image recognition tasks mainly image classification 

For more about AlexNet and CNNs architectures:

* [AlexNet paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* [CNNs Architectures, CS231n](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9)
