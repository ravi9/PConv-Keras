# Partial Convolution based Padding using Keras
Keras implementation of Partial Convolution based Padding, https://arxiv.org/pdf/1811.11718.pdf

# Dependencies
* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12

# How to use this repository

```python

  # typical convolution layer with zero padding
  x = K.conv2d(x, kernel_size=(3, 3), activation="relu",
              padding="same",
              kernel_initializer="he_uniform")
  
  # partial convolution based padding
  from libs.pconv_layer import PConv2D  
  x = PConv2D(x, kernel_size=(3, 3), activation="relu",
              padding="same",
              kernel_initializer="he_uniform")

```
