*Image: ImageNet Challenge, 2010-2017, CS231n.* 
### What's in here:

* [1. Introduction](#1)
* [2. VGG Architecture](#2)
* [3. VGG Implementation](#3)
* [4. Final Notes](#4)
<a name='1'></a>
## 1. Introduction
AlexNet was the product of extensive experimentation and refinement. Consequently, it was deficient in certain design principles. Nonetheless, the concept of stacking several layers inspired the subsequent construction of convolutional neural network architectures. 

The VGG network won in the 2014 ImageNet competition for classification and localization tasks. VGG established new ideas for the design of ConvNets, many of which remain standard today, and it possessed greater depth than AlexNet. This pattern demonstrated that deep networks consistently outperform shallow networks.

<a name='2'></a>

## 2. VGG Architecture
The VGG architecture comprises 16 layers (VGG-16) and 19 layers (VGG-19). This quantity is double the number of layers in AlexNet.

The key design tenets of the VGG network are outlined below: 

Each convolutional layer employs a kernel size of 3x3 and utilizes zero padding to ensure that the output maintains the same height and width as the input. The stride is configured to one.
* Convolutional layers are arranged in stages, with each level succeeded by a pooling layer.
* All convolutional layers employ the ReLU activation function.
* All max pooling layers utilize a pooling size of 2 and a stride of 2.
* Subsequent to the pooling layer, the ensuing convolutional step featured double the amount of filters compared to the preceding stages. If the initial convolutional layer comprises 64 filters, the subsequent layer will consist of 128 filters. This design concept is still extensively utilized today in the development of convolutional neural networks. The initial convolutional stage comprises 64 filters, the second stage contains 128 filters, the fourth stage includes 256 filters, and the fifth stage consists of 512 filters. 

The convolutional layers in VGG-16 are structured as follows:    

* Stage 1: conv - conv - pool
* Stage 2: conv - conv - pool
* Stage 3: conv - conv - conv - pool
* Stage 4: conv - conv - conv - pool
* Stage 5: conv - conv - conv - pool

In VGG-19, the 4th and 5th stages have 4 convolutional layers. 

Same as AlexNet, VGG has also 3 fully connected layers with exact same configurations: 4096 units in first two fully connected layers, 
and 1000 units in last connected layers. The last layer has a softmax activation function for classification purpose.
*Image: AlexNet and VGG-16 & VGG-19 architectures. Image captured from CS231n.*
![VGG](../images/VGG.jpg)

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
The VGG-16 network, trained on the ImageNet dataset, contains about 138 million parameters. The network is quite extensive.
<a name="4"></a>

## 4. Concluding Remarks
The VGG network is among the pioneering architectures of ConvNets that established architectural principles for visual recognition systems; yet, it possesses a substantial number of parameters, rendering it computationally inefficient. The subsequent architectures we shall examine, including GoogLeNet, tackled the difficulty of creating efficient designs suitable for mobile devices.
