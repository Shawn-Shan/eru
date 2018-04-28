# Contributing to Atom
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to ERU. 

## Ways to get involved: 
1. [Join ERU slack channel](https://join.slack.com/t/eru-framework/shared_invite/enQtMzU0NjY2NjI0NTYxLTAzN2YwOThmNzQ1MzBjZDU4MWRhMjFjMzNmNTkxZDMzZDIxZGQzZWZiNmE0MzI0MzVjN2ZhMWNiYWJiMzI4OTI)
2. Or email me at shansixioing@uchicago.edu for questions

## What should I know before I get started?
### Contribute to the backend code
1. Experienced in Python
2. Have a deep understanding of pytorch and some high-level frameworks such as Keras
### Contribute to examples and documentation
1. Understand basics of ERU backend packages
2. Experienced in some use cases of deep learning

## What should I work on now?
### Backend Code
#### Convolution Layers API
The current ERU package implemented basic layers (Dense, Dropout, Softmax, Sigmoid..) and some advanced layers (GRU, Embeddings). But we need much more layers like Convolution, Pooling, RNN, LSTM... If you are interested in implementing those layers, please look into how we wrote other layers and use the base Layer class to implement. 
Since we are still testing the general framework and might change the implementation of layers, we suggest adding only popular layers for now. 

#### Add optimizers and loss functions
We only implemented Adam optimizer and cross-entropy loss functions. Despite those being the popular ones, we need to add other optimizers and loss functions. 

#### Upgrade to Pytorch 0.4.0
There are some nice updates on Pytorch side, and we need to upgrade as well. 

#### CPU support
To save time during the development, we the code only support GPU. We need to add CPU support as well. 

#### Data Pipeline
We have struggled with the data pipeline for some time and the current implementation is similar to Keras, where users can dump x, y into the model API or write a generator. We want to support some preprocessing and ready to use generators. 
If you have experience in numpy and data processing, you can start to write some data processing code. 

### Write Examples
Given we are still working on the backend code, We suggest not add to many examples for now. But here are some examples we envision to have:
1. MNIST with MLP
2. MNIST with CNN
3. IMBD/Yelp sentiment classification with LSTM
4. IMBD/Yelp sentiment classification with Conv1D
5. Language modeling on PTB dataset
6. Sequence to sequence translation
