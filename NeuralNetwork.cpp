#include "NeuralNetwork.h"
#include <iostream>
#include <ostream>
#include <fstream>
#include <filesystem>
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

void NeuralNetwork::SaveNetwork(const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open path: " << filePath << std::endl;
        return;
    }

    file << "Epoch(s)\n";
    file << currentEpoch << "\n";
    file << "W1\n";
    for (const auto& row : W1) {
        for (const double w : row)
            file << w << " ";
        file << "\n";
    }
    file << "b1\n";
    for (const double b : b1)
        file << b << " ";
    file << "\n";
    file << "W2\n";
    for (const auto& row : W2) {
        for (const double w : row)
            file << w << " ";
        file << "\n";
    }
    file << "b2\n";
    for (const double b : b2)
        file << b << " ";
    file << "\n";

    file.close();
    std::cout << "[INFO] Neural network saved in: " << filePath << std::endl;
}

void NeuralNetwork::LoadNetwork(const std::string& filePath) {
    if (!std::filesystem::exists(filePath)) {
        std::cerr << "[ERROR] Could not find file: " << filePath << std::endl;
        return;
    }

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open path: " << filePath << std::endl;
        exit(-1);
    }

    std::string tag;
    while (file >> tag) {
        if (tag == "Epoch(s)") {
            file >> currentEpoch;
        }
        if (tag == "W1") {
            for (auto& row : W1)
                for (double& w : row)
                    file >> w;
        }
        else if (tag == "b1") {
            for (double& b : b1)
                file >> b;
        }
        else if (tag == "W2") {
            for (auto& row : W2)
                for (double& w : row)
                    file >> w;
        }
        else if (tag == "b2") {
            for (double& b : b2)
                file >> b;
        }
    }

    file.close();
    std::cout << "[INFO] Neural network loaded from: " << filePath << std::endl;
    std::cout << "[INFO] Loaded neural network currently has " << currentEpoch << " epochs!" << std::endl;
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
        totalLoss /= static_cast<double>(X.size());
        currentEpoch++;
        std::cout << "Epoch: " << epoch + 1 << " / " << epochs << " | Loss: " << totalLoss << std::endl;

        SaveNetwork("neural_network_save");
    }
}