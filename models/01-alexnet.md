
<a name='0'></a>

# ConvNets Architectures - AlexNet
Most ConvNets architectures that we will cover were the results of the [ImageNet Challenge](https://image-net.org/challenges/LSVRC/index.php) that happened from 2010 to 2017, but the most exciting things happened between 2012-2017 and that's exactly what we are going to dive deep into.

Below image give a high-level overview of the revolution of the CNNs architectures on ImageNet Challenge.
Most ConvNets architectures that we will cover were the results of the [ImageNet Challenge](https://image-net.org/challenges/LSVRC/index.php) that happened from 2010 to 2017, but the most exciting things happened between 2012-2017 and that's exactly what we are going to dive deep into.

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

AlexNet was very much similar to [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) that was introduced earlier before in 1990s except that it was deeper, and instead of using sigmoid activation functions in hidden layers like LeNet-5, it used ReLU.

Also, to reduce overfitting, AlexNet was the first paper to use dropout regularization technique. Up to day, dropout is one of the most effective regularization technique.
