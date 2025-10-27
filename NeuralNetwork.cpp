#include "NeuralNetwork.h"
#include <iostream>
#include <ostream>
#include <random>

NeuralNetwork::NeuralNetwork(const int inputSize, const int hiddenSize, const int outputSize) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution dist(-1.0, 1.0);

    W1.resize(hiddenSize, std::vector<double>(inputSize));
    W2.resize(outputSize, std::vector<double>(hiddenSize));
    b1.resize(hiddenSize);
    b2.resize(outputSize);

    for (auto& row : W1) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
    for (auto& row : W2) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
}

double NeuralNetwork::sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}
double NeuralNetwork::sigmoidDerivative(const double x) {
    return x * (1.0 - x);
}

std::vector<double> NeuralNetwork::FeedForward(const std::vector<double>& input) const {
    std::vector<double> hidden(b1.size());
    std::vector<double> output(b2.size());

    for (int i = 0; i < hidden.size(); i++) {
        double sum = b1[i];
        for (int j = 0; j < input.size(); j++) {
            sum += W1[i][j] * input[j];
        }
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < output.size(); i++) {
        double sum = b2[i];
        for (int j = 0; j < hidden.size(); j++) {
            sum += W2[i][j] * hidden[j];
        }
        output[i] = sigmoid(sum);
    }
    return output;
}

void NeuralNetwork::TrainNetwork(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, const double learningRate, const int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        for (int n = 0; n < X.size(); n++) {
            const std::vector<double>& input(X[n]);
            const std::vector<double>& target(Y[n]);

            std::vector<double> hidden(b1.size());
            std::vector<double> output(b2.size());
            for (int i = 0; i < hidden.size(); i++) {
                double sum = b1[i];
                for (int j = 0; j < input.size(); j++) {
                    sum += W1[i][j] * input[j];
                }
                hidden[i] = sigmoid(sum);
            }
            for (int i = 0; i < output.size(); i++) {
                double sum = b2[i];
                for (int j = 0; j < hidden.size(); j++) {
                    sum += W2[i][j] * hidden[j];
                }
                output[i] = sigmoid(sum);
            }

            std::vector<double> outputError(output.size());
            for (int i = 0; i < output.size(); i++) {
                outputError[i] = target[i] - output[i];
                totalLoss += 0.5 * std::pow(outputError[i], 2);
            }
            std::vector<double> outputDelta(output.size());
            for (int i = 0; i < output.size(); i++) {
                outputDelta[i] = outputError[i] * sigmoidDerivative(output[i]);
            }

            std::vector hiddenError(hidden.size(), 0.0);
            for (int i = 0; i < hidden.size(); i++) {
                for (int j = 0; j < output.size(); j++) {
                    hiddenError[i] += W2[j][i] * outputDelta[j];
                }
            }
            std::vector<double> hiddenDelta(hidden.size());
            for (int i = 0; i < hidden.size(); i++) {
                hiddenDelta[i] = hiddenError[i] * sigmoidDerivative(hidden[i]);
            }

            for (int i = 0; i < output.size(); i++) {
                for (int j = 0; j < hidden.size(); j++) {
                    W2[i][j] += learningRate * outputDelta[i] * hidden[j];
                }
                b2[i] += learningRate * outputDelta[i];
            }

            for (int i = 0; i < hidden.size(); i++) {
                for (int j = 0; j < input.size(); j++) {
                    W1[i][j] += learningRate * hiddenDelta[i] * input[j];
                }
                b1[i] += learningRate * hiddenDelta[i];
            }
        }
        totalLoss /= X.size();
        std::cout << "Epoch: " << epoch + 1 << " / " << epochs << " | Loss: " << totalLoss << std::endl;
    }
}