# Digit Recognition Artificial Intelligence

## General
Within a day, I created this AI as a school project in computer science.
This AI is able to recognize handwritten digits ranging from 0 to 9 with
60.000 worth of training data from the MNIST database and 10.000 test data.

## The AI
The neural network consist of sigmoid neurons used in the early days of 
past AIs. There is 1 input layer, hidden layer and output layer each.
The input layer contains 784 neurons each for one pixel from a 28x28 grayscale image of the digit
ranging from 0 to 1 (black to white). The hidden layer only has 64 and
the output layer 10 neurons each one representing the digit 0, 1, 2, 3, ..., 9.
Here I will explain my progress and how I made this AI in C++ as well as how a neural network kinda works

## How does an AI work?
Let us take the example from my digit recognition AI.
Before that, this is the sigmoid function:

`o(x) = 1 / ( 1 + e^-x)`

As mentioned before, every AI (at least the ones still using sigmoid functions and old structures)
is like this or similar, the only thing may differ is the amount of hidden layers where you can use at least 1.
To understand what I am trying to explain you should know at least what a perceptron is.
The sigmoid neuron is quite similar one of the differences is as follows:

Perceptron: `w1 * x1 + w2 * x2 + wn * xn + ... >= theta (if true result is 1 else 0)`
Sigmoid neuron: `result = o(w1 * x1 + w2 * x2 + wn * xn + ... + b)`

As you can see, the sigmoid neuron does not have theta at all. You could say that b is actually our theta
but instead is not used for comparison but as a part of the calculation. Also compared to the perceptron, this neuron
does not have a big impact on the result if the weights and biases change a little. The sigmoid function just makes sure that
the result should be between 0 and 1. And that is roughly what a sigmoid neuron is.

The neural network just has mulitple neurons.The input neurons just gets the data in that case the pixel color value, the hidden
ones do some calculations depending on the biases (b) and weights (w1, w2, wn, ... ) and the output neurons well they contain 
the final value
