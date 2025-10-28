#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <ostream>
#include <fstream>
#include <filesystem>
#include <random>
#include "Timer.h"

NeuralNetwork::NeuralNetwork(const int inputSize, const int hiddenSize, const int outputSize) : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    W1.resize(hiddenSize, std::vector<float>(inputSize));
    W2.resize(outputSize, std::vector<float>(hiddenSize));
    b1.resize(hiddenSize);
    b2.resize(outputSize);

    dW1.resize(hiddenSize, std::vector<float>(inputSize));
    dW2.resize(outputSize, std::vector<float>(hiddenSize));
    dB1.resize(hiddenSize);
    dB2.resize(outputSize);

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

void NeuralNetwork::ResetGradients() {
    for (auto& row : dW1) std::ranges::fill(row, 0.0f);
    for (auto& row : dW2) std::ranges::fill(row, 0.0f);
    std::ranges::fill(dB1, 0.0f);
    std::ranges::fill(dB2, 0.0f);
}

void NeuralNetwork::AccumulateGradient(const std::vector<float>& X, const std::vector<float>& Y) {
    const auto& input = X;
    const auto& target = Y;

    std::vector<float> hidden(hiddenSize);
    std::vector<float> output(outputSize);

    for (int h = 0; h < hiddenSize; h++) {
        float sum = b1[h];
        for (int i = 0; i < inputSize; i++)
            sum += W1[h][i] * input[i];
        hidden[h] = sigmoid(sum);
    }

    for (int o = 0; o < outputSize; o++) {
        float sum = b2[o];
        for (int h = 0; h < hiddenSize; h++)
            sum += W2[o][h] * hidden[h];
        output[o] = sigmoid(sum);
    }

    std::vector<float> deltaOut(outputSize);
    for (int o = 0; o < outputSize; o++) {
        deltaOut[o] = (output[o] - target[o]) * sigmoidDerivative(output[o]);
    }
    std::vector deltaHid(hiddenSize, 0.0f);
    for (int h = 0; h < hiddenSize; h++) {
        float sum = 0;
        for (int o = 0; o < outputSize; o++)
            sum += deltaOut[o] * W2[o][h];
        deltaHid[h] = sum * sigmoidDerivative(hidden[h]);
    }

    for (int o = 0; o < outputSize; o++) {
        for (int h = 0; h < hiddenSize; h++) {
            dW2[o][h] += deltaOut[o] * hidden[h];
        }
    }
    for (int h = 0; h < hiddenSize; h++) {
        for (int i = 0; i < inputSize; i++) {
            dW1[h][i] += deltaHid[h] * input[i];
        }
    }
    for (int o = 0; o < outputSize; o++) {
        dB2[o] += deltaOut[o];
    }
    for (int h = 0; h < hiddenSize; h++) {
        dB1[h] += deltaHid[h];
    }
}

void NeuralNetwork::ApplyGradient(const int batchSize, const float learningRate) {
    const float scale = learningRate / static_cast<float>(batchSize);

    for (int h = 0; h < hiddenSize; h++) {
        for (int i = 0; i < inputSize; i++) {
            W1[h][i] -= scale * dW1[h][i];
        }
    }
    for (int o = 0; o < outputSize; o++) {
        for (int h = 0; h < hiddenSize; h++) {
            W2[o][h] -= scale * dW2[o][h];
        }
    }
    for (int h = 0; h < hiddenSize; h++) {
        b1[h] -= scale * dB1[h];
    }
    for (int o = 0; o < outputSize; o++) {
        b2[o] -= scale * dB2[o];
    }
}

void NeuralNetwork::TrainNetwork(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, const float learningRate, const int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        constexpr int batchSize = 64;

        for (int n = 0; n < X.size(); n += batchSize) {
            ResetGradients();
            const int realBatchSize = std::min(batchSize, static_cast<int>(X.size()) - n);
            for (int b = 0; b < realBatchSize; b++) {
                AccumulateGradient(X[n + b], Y[n + b]);
            }
            ApplyGradient(realBatchSize, learningRate);
        }

        currentEpoch++;
        std::cout << "Epoch: " << epoch + 1 << " / " << epochs << std::endl;
        SaveNetwork("neural_network_save");
    }
}