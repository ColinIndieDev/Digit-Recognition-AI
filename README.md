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

### Sigmoid neurons and neural network
As mentioned before, every AI (at least the ones still using sigmoid functions and old structures)
is like this or similar, the only thing may differ is the amount of hidden layers where you can use at least 1.
To understand what I am trying to explain you should know at least what a perceptron is.
The sigmoid neuron is quite similar one of the differences is as follows:

Perceptron: `w1 * x1 + w2 * x2 + wn * xn + ... >= theta (if true result is 1 else 0)`\
Sigmoid neuron: `result = o(w1 * x1 + w2 * x2 + wn * xn + ... + b)`

As you can see, the sigmoid neuron does not have theta at all. You could say that b is actually our theta
but instead is not used for comparison but as a part of the calculation. Also compared to the perceptron, this neuron
does not have a big impact on the result if the weights and biases change a little. The sigmoid function just makes sure that
the result should be between 0 and 1. And that is roughly what a sigmoid neuron is.

The neural network just has multiple neurons which are connected to the next layer of neurons.The input neurons just gets the data in that case the pixel color value, the hidden
ones do some calculations depending on the biases (b) and weights (w1, w2, wn, ... ) and the output neurons well they contain 
the final value in that case the probability what number the image shows.

### Backpropagation
But before we can test our neural network, we have to train it before to increase its accuracy. Here I used supervised machine learning where the train images are labeled with the correct digit the picture shows.
This process or a part of it of the training is "backpropagation" when we start from the output to the input.
The first step is to calculate the errors for each output neuron:

`std::vector<double> outputError(output.size());`\
`for (int i = 0; i < output.size(); i++) {`\
`   outputError[i] = target[i] - output[i];`\
`}`

`output` is the result the AI gets and `target` the expected result the AI should get as well. As you can see we just subtract both of them to the the errors from the output.

The next step is to calculate the delta of output which means the factor we want to change the weights and biase from the output neuron. We can achieve this like this:

`std::vector<double> outputDelta(output.size());`\
`for (int i = 0; i < output.size(); i++) {`\
   `outputDelta[i] = outputError[i] * sigmoidDeriv(output[i]);`\
`}`

Here we just multiply the error with the sigmoid derivative which is:
`o'(o(x)) = o(x) * (1.0 - o(x))`

Now we do the similar process for the hidden neurons. We just have different inputs for the functions in contrast to the output. But they both do the same thing: changing the weights and biases for each neuron.
Note that we only calculated the change which must happen but we now need to apply this:

`for (int i = 0; i < output.size(); i++) {`\
`    for (int j = 0; j < hidden.size(); j++) {`\
`        W2[i][j] += learningRate * outputDelta[i] * hidden[j];`\
`    }`\
`    b2[i] += learningRate * outputDelta[i];`\
`}`\
`for (int i = 0; i < hidden.size(); i++) {`\
`    for (int j = 0; j < input.size(); j++) {`\
`        W1[i][j] += learningRate * hiddenDelta[i] * input[j];`\
`    }`\
`    b1[i] += learningRate * hiddenDelta[i];`\
`}`

We firstly mulitply our calculated delta with the result of the neuron and the learningRate where the rate opf 0.1 is recommended. Then we add this to the weight.
For the baise we only multiply this with the delta and learning rate and add the result to the bias.

This way, we roughly can train our network of course there is a bit more behind the scenes but backpropagation is the most important thing inside the train function and also one of the difficult things when creating a network.

