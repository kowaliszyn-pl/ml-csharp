# Neural Networks Examples

This project shows the examples how to create and train neural networks using the NeuralNetworks library (https://github.com/kowaliszyn-pl/ml-csharp/tree/master/src/NeuralNetworks).

It contains the following examples:

- **Function data set** - demonstrates how to create and train a neural network to approximate a mathematical function.
- **Boston Housing data set** - demonstrates how to create and train a neural network to predict housing prices in Boston:
	- using custom model,
	- using generic model
- **MNIST data set** - demonstrates how to create and train a neural network to recognize handwritten digits:
	- using dense layers,
	- using CNN layers.


## MNIST Results

The following results were obtained on the MNIST data set using dense layers for 12 epochs:

| # | Learning low | Learning high | I layer activation | II layer activation | I dropout keep prob | II dropout keep prob | Loss                   | Optimizer              | Epochs | Eval on test |
|---|--------------|---------------|---------|----------|-----------|------------|------------------------|------------------------|--------|--------------|
| 1 | 0,005        | 0,0009        | Sigmoid | Sigmoid  | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 86,03%       |
| 2 | 0,005        | 0,0009        | Tanh    | Sigmoid  | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 91,57%       |
| 3 | 0,005        | 0,0009        | Sigmoid | Tanh     | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 91,84%       |
| 4 | 0,005        | 0,0009        | Tanh    | Tanh     | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 94,25%       |
| 5 | 0,005        | 0,0009        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 95,20%       |
| 6 | 0,005        | 0,0009        | ReLU 1  | Sigmoid  | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 92,34%       |
| 7 | 0,005        | 0,0009        | ReLU 1  | ReLU 1   | 0,85      | 0,85       | softmax cross entropy | GDMomentum 0,9         | 12     | 95,19%       |
| 8 | 0,005        | 0,0009        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,11%       |
| 9 | 0,004        | 0,0008        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                 | 12     | 97,15%       |
|10 | 0,005        | 0,0008        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,06%       |
|11 | 0,004        | 0,0009        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,04%       |
|12 | 0,0035       | 0,00075       | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,23%       |
|13 | 0,003        | 0,0007        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,31%       |
|14 | 0,002        | 0,0006        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,31%       |
|15 | 0,002        | 0,0007        | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90                   | 12     | 97,28%       |
|16 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,90        | 12     | 97,38%       |
|17 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,85        | 12     | 97,32%       |
|18 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,91        | 12     | 97,24%       |
|19 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,41% *  |
|20 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,84      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,35%       |
|21 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,86      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,39%       |
|22 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,87      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,26%       |
|23 | 0,0025       | 0,00065       | ReLU 1  | Tanh     | 0,83      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,30%       |
|24 | 0,0025       | 0,00065       | ReLU 1  | ReLU 0,5     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,28%       |
|25 | 0,0025       | 0,00065       | ReLU 1  | ReLU 1     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,18%       |
|26 | 0,0025       | 0,0005       | ReLU 1  | Tanh     | 0,85      | 0,85       | softmax cross entropy | Adam, beta 0,89        | 12     | 97,18%       |

MNIST CNN

Runs 

1. Epochs: 15, Learning rate: 0.01 - 0.001, Dropout keep prob: 0,85, Optimizer: GDMomentum 0,9, Activations: Tanh4D, Filters: 32, Kernel size: 3

Accuracy: 