Assignment for the undergraduate course Machine Intelligence (UE18CS303)

Team PES1201800395/PES1201801549/PES1201801618

Design of a Neural network from scratch 

Neural network architecture: LINEAR->SIGMOID->LINEAR->SIGMOID

Hyperparameters used:-
1) Number of layers = 2
2) Number of neurons/layer = input layer:8 hidden layer:17, output layer:1
3) Dimensions of weight and bias matrices = W1:(17, 8) b1:(17, 1) W2:(1, 17) b2:(1, 1)
4) Activation functions used in each layer = Sigmoid
5) Loss function = Binary cross entropy
6) Optimization = Adam optimization
7) Learning rate = 0.001
8) First moment (beta 1) = 0.9
9) Second moment (beta 2) = 0.999
10) Epsilon value = 10e-05
11) Number of iterations used to train the model = 30000

Implementation:-
We have designed a two-layer neural network from scratch using a vectorised implementation and NumPy module. Python dictionaries were made use of extensively
to store and return values while initialising parameters, updating parameters, during forward and backward propagation. Adam algorithm is also implemented to
optimize cost. This is visualized by plotting 'cost vs iterations'. 

Key features implemented:-
1) Random initialization (to weights and biases matrices): this speeds up the process of optimization. 
2) Vectorised implementation: program runs faster due to parallelism.
3) Adam optimization: it is faster than gradient descent.
4) Plot of 'cost vs iterations': helps to visualize the cost optimization. This plot can also be used for Early stopping regularization method. 

Beyond the basics implementations:-
1) Adam optimization algorithm
2) 'cost vs iterations' plot

Steps to run files:-
1) python pre-processing.py
2) python neural-net.py
