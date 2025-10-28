#pragma once
#include <vector>
#include <string>

class mnist_loader {
public:
    static int reverse_int(int i);
    static std::vector<std::vector<float>> load_images(const std::string &filename);
    static std::vector<int> load_labels(const std::string &filename);
};