# Digit Recognition Artificial Intelligence

## General
Within a day, I created this AI as a school project in computer science.
This AI is able to recognize handwritten digits ranging from 0 to 9 with
60.000 worth of training data from the MNIST database and 10.000 test data.

## My AI
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

You can look up the graph online but the key of the sigmoid function is that:

`lim (x -> -∞) o(x) = 0`\
`lim (x -> +∞) o(x) = 1`

That basically means that the result is "clamped" between 0 and 1 where x can me ANY number. For high positive values the result gets
closer to 1 and low negative values to 0!

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

The neural network just has multiple neurons which are connected to the next layer of neurons. The input neurons just get the data in that case the pixel color value while the hidden and output
ones do some calculations depending on the biases (b) and weights (w1, w2, wn, ... ) and the output neurons well they after contain 
the final value in that case the probability what number the image shows.

### Backpropagation
But before we can test our neural network, we have to train it before to increase its accuracy. Here I used supervised machine learning where the train images are labeled with the correct digit the picture shows.
This process or a part of it of the training is "backpropagation" when we start from the output to the input.
The first step is to calculate the errors for each output neuron if for example the image shows a "5" but the AI thinks it is a "8" with 0.8 probablity while 5 only has 0.3. We have to change our weights and biases since these are the only things we can change and adjust so that for 5 the result is 1.0 but for 8 0.0:

`std::vector<double> outputError(output.size());`\
`for (int i = 0; i < output.size(); i++) {`\
`   outputError[i] = target[i] - output[i];`\
`}`

`output` is the result the AI gets and `target` the expected result the AI should get as well. As you can see we just subtract both of them to get the errors from the output.

The next step is to calculate the delta of output which means the factor we want to change the weights and biases from the output neuron. We can achieve this like this:

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

We firstly multiply our calculated delta with the result of the neuron and the `learningRate` where the rate of 0.1 is recommended. Then we add this to the weight.
For the biases we only multiply each with the delta and learning rate and add the result to the bias.

This way, we roughly can train our network of course there is a bit more behind the scenes but backpropagation is the most important thing inside the training function and also one of the difficult things when creating a network like that.

### Forwardpropagation

Now we have successfully trained our AI. This can be done by the provided function TrainNetwork(...). Here we use as parameters the train images and the corresponding labels from the MNIST database, how many epochs we want the neural network to train and the learn rate:

`NeuralNetwork network(784, 64, 10);`\
`network.TrainNetwork(trainImages, Y, 0.1, 50);`

Here we create an instance of our neural network before and set the amount of neurons in the input, hidden and output layer. When training we refer to the train images `trainImages` and train labels `Y`. Additionally the learn rate is set to `0.1` and we want the training to go until `50` epochs.

That is enough for now because we want to test our AI and how good it is. We measure its performance by "feeding" our neural network with the test data (10.000 images). Here we need "forwardpropagation".
Forwardpropagation is used to get the results from our 10 output neurons. First we already converted our 28x28 grayscale image into 28 different vectors each of the size of 28 containing the grayscale value between 0 and 1. Then we calculate the result of our hidden neurons like this:

`for (int i = 0; i < hidden.size(); i++) {`\
`   float sum = b1[i];`\
`   for (int j = 0; j < input.size(); j++) {`\
`       sum += W1[i][j] * input[j];`\
`   }`\
`   hidden[i] = sigmoid(sum);`\
`}`

Here we basically iterate through all hidden neurons. For each neuron we do:

`hidden = sigmoid(w1*x1 + w2*x2 + w3*x3 + wn*xn + ... + b)`

We add the product of the value the hidden neuron gets from the input neuron with the corresponding weight for the input neuron's result and finally add our bias. In that case we have 784 input neurons so we add 784 products of weight and value.
The bias is for each hidden neuron different we have 64 of them since we have 64 hidden neurons.

We literally do the same thing for the output neurons as well so it will be something like this for each output neuron:

`output = sigmoid(w1*x1 + w2*x2 + w3*x3 + wn*xn + ... + b)`

Here is the only difference that we have 64 hidden neurons so 64 products of weight and value since the output neurons gets his input data from the hidden neurons. We have 10 output neurons so obviously we have 10 biases for each output neuron.

And all of this I mentioned is forwardpropagation. Compared to backpropagation we start from the input neurons with the pixel data and fire the data to all hidden neurons and the hidden neurons to the output neurons which do the final calculation until we get the "probability" which number is likely shown on the grayscale image.

> [!IMPORTANT]
> For conclusion, backpropagation is used to train our neural network with train data
> and forwardpropagation to test our neural network with test data or to solve a new
> image showing a digit

## Future plans
This example of an AI is outdated today. These days we have even more efficient AIs. The difference are the functions being used instead of sigmoid like ReLu, tanh etc. Also the structure is a bit different when you compare this AI with other ones nowadays or LLMs like ChatGPT. But this project is a great start when trying to understand and even create neural networks. We can optimize this here I used "batching" so that we don't iterate through each train image during backpropagation we instead take only a chunk of images each iteration so the AI calculates the changes delta for the weights and biases for 64 images and after that it takes another chunk and so on until we went through all 60.000 training images. This saves us multiple seconds but we can optimize it even more!
From here you should be able to make such a primitive AI on your own and upgrade it. If you don't understand or need some help I would recommend this playlist on YouTube especially the first 3 videos which also helped me much and made me do this AI on a single day. You can also check out this "book":

### Sources:
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1 (neural network)

https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2 (deepening)

https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3 (backpropagation)

http://neuralnetworksanddeeplearning.com/chap1.html ("book" also about digit recognition AI with sigmoid)
