*Image: ImageNet Challenge, 2010-2017, CS231n.* 
### What's in here:

* [1. Introduction](#1)
* [2. VGG Architecture](#2)
* [3. VGG Implementation](#3)
* [4. Final Notes](#4)
<a name='1'></a>
## 1. Introduction
AlexNet was the results of lots of trials and errors. As a consequence, it lacked some design principles. But despite that, the idea of stacking layers after layers inspired
the next generation of convolutional neural network architectures. 

[VGG](https://arxiv.org/pdf/1409.1556.pdf) network won the ImageNet competition of 2014 in classification and localization tasks. VGG introduced new principles for 
designing ConvNets (some of which are still norms to day), and it was deeper than AlexNet. Such trend proved that deep networks always perform better than shallow networks.

<a name='2'></a>

## 2. VGG Architecture
In total, [VGG](https://arxiv.org/pdf/1409.1556.pdf) architecture has 16 layers(VGG-16) and 19 layers(VGG-19). That's two times the number of layers of AlexNet.

Below are the main design principles of VGG network: 

* Every single convolutional layer has a kernel size of 3X3 and it used zero padding so that the output has the same height and width as the input. The stride is also set to 1.
* Convolutional layers are stacked in stages, each stage followed by a pooling layer.
* All convolutional layers use ReLU activation function.
* All maxpooling layers have a pooling size of 2 and stride of 2. 
* After the pooling layer, the next convolutional stage had double the number of filters from the previous stages. So if the first convolutional stage was 64, the next will be 128. This design idea is still used alot to day in designing convolutional neural networks. To be specific here, the first convolutional stage has 64 filters, second 128, fourth 256, fifth 512 filters. 

The convolutional stages in VGG-16 are organized as follows:    

* Stage 1: conv - conv - pool
* Stage 2: conv - conv - pool
* Stage 3: conv - conv - conv - pool
* Stage 4: conv - conv - conv - pool
* Stage 5: conv - conv - conv - pool

In VGG-19, the 4th and 5th stages have 4 convolutional layers. 

Same as AlexNet, VGG has also 3 fully connected layers with exact same configurations: 4096 units in first two fully connected layers, 
and 1000 units in last connected layers. The last layer has a softmax activation function for classification purpose.
*Image: AlexNet and VGG-16 & VGG-19 architectures. Image captured from CS231n.*
[]("./images/VGG.jpg)
Justin Johnson has also a great side by side comparison of AlexNet and VGG in his Deep Learning for Computer Vision course.
<a name='3'></a>

## 3. VGG Implementation
As we now understand the ins and out of the VGG network, let's implement it. There are many versions of VGG such as VGG-11, VGG-16, VGG-19. We will implement VGG-16 but
the process is similar to how you would implement other versions. 

A quickiest way to implement VGG is to just stack all layers with their respective hyperparameters, block after a block. A fancier and clean way would be to build a
convolutional block as a function that takes number of convolutional layers and filters and reuse
it for each block.

```python
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        # Define convolutional blocks (features)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers (classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate the model
vgg_16 = VGG16(num_classes=1000)
```
The VGG-16 network trained on Imagenet dataset has over 138 millions parameters. That's a pretty big network.
<a name='4'></a>

## 4. Final Notes
VGG network is one of the first ConvNets architectures that introduced some design principles in designing visual recognition architectures 
but it has lots of parameters which is not a computational efficient. The next architectures that we will cover such as GoogLeNet addressed the 
challenge of designing efficient architectures that can also run in mobile devices.
