#pragma once
#include <string>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    void LoadNetwork(const std::string& filePath);
    void SaveNetwork(const std::string& filePath) const;
    [[nodiscard]] std::vector<double> FeedForward(const std::vector<double>& input) const;
    void TrainNetwork(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, double learningRate, int epochs);
private:
    static double sigmoid(double x);
    static double sigmoidDerivative(double x);

    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b1;
    std::vector<double> b2;
    int currentEpoch = 0;
};
