#include <iostream>
#include "NeuralNetwork.h"
#include "mnist_loader.h"
#include <chrono>

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> duration{};
    std::string text;

    explicit Timer(const std::string& text) : text(text) {
        start = std::chrono::steady_clock::now();
    }

    ~Timer() {
        end = std::chrono::steady_clock::now();
        duration = end - start;
        const float ms = duration.count() * 1000.0f;
        std::cout << "------------------------------------------------------\n";
        std::cout << text.c_str() << " " << ms << " ms!\n";
        std::cout << "------------------------------------------------------\n";
    }
};

int main() {
    const auto trainImages = mnist_loader::load_images("../data/train-images.idx3-ubyte");
    const auto trainLabels = mnist_loader::load_labels("../data/train-labels.idx1-ubyte");
    const auto testImages = mnist_loader::load_images("../data/t10k-images.idx3-ubyte");
    const auto testLabels = mnist_loader::load_labels("../data/t10k-labels.idx1-ubyte");

    std::vector<std::vector<double>> Y;
    for (const int label : trainLabels) {
        std::vector oneHot(10, 0.0);
        oneHot[label] = 1.0;
        Y.push_back(oneHot);
    }

    NeuralNetwork network(784, 64, 10);
    {
        auto timer = Timer("Training network took");
        network.TrainNetwork(trainImages, Y, 0.1, 20);
    }
    {
        auto timer = Timer("Testing network took");

        int correct = 0;
        const int total = testImages.size();

        for (int i = 0; i < total; i++) {
            auto output = network.FeedForward(testImages[i]);

            const int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

            if (const int actual = testLabels[i];
                predicted == actual) correct++;
        }

        const double accuracy = 100.0 * correct / total;
        std::cout << "Test accuracy: " << accuracy << "% (" << correct << "/" << total << ")" << std::endl;
    }
}
