#pragma once
#include <string>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    void LoadNetwork(const std::string& filePath);
    void SaveNetwork(const std::string& filePath) const;
    [[nodiscard]] std::vector<float> FeedForward(const std::vector<float>& input) const;
    void TrainNetwork(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, float learningRate, int epochs);
private:
    static float sigmoid(float x);
    static float sigmoidDerivative(float x);

    void ResetGradients();
    void AccumulateGradient(const std::vector<float>& X, const std::vector<float>& Y);
    void ApplyGradient(int batchSize, float learningRate);

    std::vector<std::vector<float>> dW1;
    std::vector<std::vector<float>> dW2;
    std::vector<float> dB1;
    std::vector<float> dB2;
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<float>> W1;
    std::vector<std::vector<float>> W2;
    std::vector<float> b1;
    std::vector<float> b2;
    int currentEpoch = 0;
};
