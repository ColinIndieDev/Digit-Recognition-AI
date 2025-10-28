#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <ostream>
#include <fstream>
#include <filesystem>
#include <random>
#include "Timer.h"

NeuralNetwork::NeuralNetwork(const int inputSize, const int hiddenSize, const int outputSize) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    W1.resize(hiddenSize, std::vector<float>(inputSize));
    W2.resize(outputSize, std::vector<float>(hiddenSize));
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
        for (const float w : row)
            file << w << " ";
        file << "\n";
    }
    file << "b1\n";
    for (const float b : b1)
        file << b << " ";
    file << "\n";
    file << "W2\n";
    for (const auto& row : W2) {
        for (const float w : row)
            file << w << " ";
        file << "\n";
    }
    file << "b2\n";
    for (const float b : b2)
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
                for (float& w : row)
                    file >> w;
        }
        else if (tag == "b1") {
            for (float& b : b1)
                file >> b;
        }
        else if (tag == "W2") {
            for (auto& row : W2)
                for (float& w : row)
                    file >> w;
        }
        else if (tag == "b2") {
            for (float& b : b2)
                file >> b;
        }
    }

    file.close();
    std::cout << "[INFO] Neural network loaded from: " << filePath << std::endl;
    std::cout << "[INFO] Loaded neural network currently has " << currentEpoch << " epochs!" << std::endl;
}

float NeuralNetwork::sigmoid(const float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
float NeuralNetwork::sigmoidDerivative(const float x) {
    return x * (1.0f - x);
}

std::vector<float> NeuralNetwork::FeedForward(const std::vector<float>& input) const {
    std::vector<float> hidden(b1.size());
    std::vector<float> output(b2.size());

    for (int i = 0; i < hidden.size(); i++) {
        float sum = b1[i];
        for (int j = 0; j < input.size(); j++) {
            sum += W1[i][j] * input[j];
        }
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < output.size(); i++) {
        float sum = b2[i];
        for (int j = 0; j < hidden.size(); j++) {
            sum += W2[i][j] * hidden[j];
        }
        output[i] = sigmoid(sum);
    }
    return output;
}

void NeuralNetwork::TrainNetwork(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, const float learningRate, const int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float totalLoss = 0.0;

        for (int n = 0; n < X.size(); n++) {
            const std::vector<float>& input(X[n]);
            const std::vector<float>& target(Y[n]);

            std::vector<float> hidden(b1.size());
            std::vector<float> output(b2.size());

            for (int i = 0; i < hidden.size(); i++) {
                float sum = b1[i];
                for (int j = 0; j < input.size(); j++) {
                    sum += W1[i][j] * input[j];
                }
                hidden[i] = sigmoid(sum);
            }
            for (int i = 0; i < output.size(); i++) {
                float sum = b2[i];
                for (int j = 0; j < hidden.size(); j++) {
                    sum += W2[i][j] * hidden[j];
                }
                output[i] = sigmoid(sum);
            }

            std::vector<float> outputError(output.size());
            for (int i = 0; i < output.size(); i++) {
                outputError[i] = target[i] - output[i];
                totalLoss += 0.5f * std::pow(outputError[i], 2.0f);
            }
            std::vector<float> outputDelta(output.size());
            for (int i = 0; i < output.size(); i++) {
                outputDelta[i] = outputError[i] * sigmoidDerivative(output[i]);
            }

            std::vector hiddenError(hidden.size(), 0.0f);
            for (int i = 0; i < hidden.size(); i++) {
                for (int j = 0; j < output.size(); j++) {
                    hiddenError[i] += W2[j][i] * outputDelta[j];
                }
            }
            std::vector<float> hiddenDelta(hidden.size());
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
        totalLoss /= static_cast<float>(X.size());
        currentEpoch++;
        std::cout << "Epoch: " << epoch + 1 << " / " << epochs << " | Loss: " << totalLoss << std::endl;

        SaveNetwork("neural_network_save");
    }
}