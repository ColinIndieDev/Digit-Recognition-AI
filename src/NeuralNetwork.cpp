#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <ostream>
#include <fstream>
#include <filesystem>
#include <float.h>
#include <random>
#include "TimerChrono.h"

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
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[N.N. SAVE] Cannot open path: " << filePath << std::endl;
        return;
    }

    int ce = currentEpoch;
    out.write(reinterpret_cast<char*>(&ce), sizeof(int));

    int is = inputSize, hs = hiddenSize, os = outputSize;
    out.write(reinterpret_cast<char*>(&is), sizeof(int));
    out.write(reinterpret_cast<char*>(&hs), sizeof(int));
    out.write(reinterpret_cast<char*>(&os), sizeof(int));

    for (int h = 0; h < hiddenSize; h++) {
        out.write(reinterpret_cast<const char*>(W1[h].data()), static_cast<std::streamsize>(inputSize * sizeof(float)));
    }
    for (int o = 0; o < outputSize; o++) {
        out.write(reinterpret_cast<const char*>(W2[o].data()), static_cast<std::streamsize>(hiddenSize * sizeof(float)));
    }

    out.write(reinterpret_cast<const char*>(b1.data()), static_cast<std::streamsize>(hiddenSize * sizeof(float)));
    out.write(reinterpret_cast<const char*>(b2.data()), static_cast<std::streamsize>(outputSize * sizeof(float)));

    out.close();
    std::cout << "[N.N. SAVE] Neural network saved in: " << filePath << std::endl;
}

void NeuralNetwork::LoadNetwork(const std::string& filePath) {
    if (!std::filesystem::exists(filePath)) {
        std::cerr << "[N.N. LOAD] Could not find file: " << filePath << std::endl;
        return;
    }

    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Cannot open path: " << filePath << std::endl;
        exit(-1);
    }

    in.read(reinterpret_cast<char*>(&currentEpoch), sizeof(int));

    int is, hs, os;
    in.read(reinterpret_cast<char*>(&is), sizeof(int));
    in.read(reinterpret_cast<char*>(&hs), sizeof(int));
    in.read(reinterpret_cast<char*>(&os), sizeof(int));

    for (int h = 0; h < hiddenSize; h++) {
        in.read(reinterpret_cast<char*>(W1[h].data()), static_cast<std::streamsize>(inputSize * sizeof(float)));
    }
    for (int o = 0; o < outputSize; o++) {
        in.read(reinterpret_cast<char*>(W2[o].data()), static_cast<std::streamsize>(hiddenSize * sizeof(float)));
    }

    in.read(reinterpret_cast<char*>(b1.data()), static_cast<std::streamsize>(hiddenSize * sizeof(float)));
    in.read(reinterpret_cast<char*>(b2.data()), static_cast<std::streamsize>(outputSize * sizeof(float)));

    in.close();
    std::cout << "[N.N. LOAD] Neural network loaded from: " << filePath << std::endl;
    std::cout << "[N.N. LOAD] Loaded neural network currently has " << currentEpoch << " epochs!" << std::endl;
}

float NeuralNetwork::sigmoid(const float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
float NeuralNetwork::sigmoidDerivative(const float x) {
    return x * (1.0f - x);
}

std::vector<std::vector<float>> NeuralNetwork::ActivationHeatMap(const std::vector<float>& input) const {
    std::vector<float> hidden(hiddenSize);

    for (int h = 0; h < hiddenSize; h++) {
        float sum = b1[h];
        for (int i = 0; i < inputSize; i++) {
            sum += W1[h][i] * input[i];
        }
        hidden[h] = sigmoid(sum);
    }

    std::vector heat(28, std::vector(28, 0.0f));
    for (int h = 0; h < hiddenSize; h++) {
        for (int i = 0; i < inputSize; i++) {
            heat[i / 28][i % 28] += std::abs(W1[h][i]) * hidden[h];
        }
    }

    float minValue;
    float maxValue;
    for (auto& row : heat) {
        for (float v : row) {
            minValue = std::min(minValue, v);
            maxValue = std::max(maxValue, v);
        }
    }
    float range = maxValue - minValue;
    if (range < 1e-6f) range = 1.0f;
    for (auto& row : heat) {
        for (float& v : row) {
            v = (v - minValue) / range;
        }
    }
    return heat;
}

std::vector<float> NeuralNetwork::RelevanceMap(const std::vector<float>& input, const int outputIndex) const {
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

    std::vector deltaOut(outputSize, 0.0f);
    deltaOut[outputIndex] = sigmoidDerivative(output[outputIndex]);

    std::vector deltaHid(hiddenSize, 0.0f);
    for (int h = 0; h < hiddenSize; h++) {
        float sum = 0.0f;
        for (int o = 0; o < outputSize; o++)
            sum += deltaOut[o] * W2[o][h];
        deltaHid[h] = sum * sigmoidDerivative(hidden[h]);
    }

    std::vector relevance(inputSize, 0.0f);
    for (int h = 0; h < hiddenSize; h++) {
        for (int i = 0; i < inputSize; i++) {
            relevance[i] += deltaHid[h] * W1[h][i];
        }
    }

    return relevance;
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
        std::cout << "[N.N. TRAINING] Epoch(s) trained: " << epoch + 1 << " / " << epochs << " (Total epochs: " << currentEpoch << ")" << std::endl;
        SaveNetwork("neural_network_save");
    }
}