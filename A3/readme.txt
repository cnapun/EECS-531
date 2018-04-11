I have a bunch of files here. As a baseline, I have logistic regresion and a 1 hidden layer (64 hidden unit) network.
As a slightly more complicated model, I have a CNN taking some inspiration from SqueezeNet, and a 10-block ResNet.
Both of those achieve better performance than the baseline, and the ResNet does by a significant margin.

DeepDream experiments with the idea of generating images to maximize response in a particular layer.
MNISTFool experiments with adversarial image generation.
model.png contains a graph depicting the ResNet model.
Weights for the CNNs and simple NN can be found in the .h5 files