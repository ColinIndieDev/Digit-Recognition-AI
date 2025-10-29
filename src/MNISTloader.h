#pragma once
#include <vector>
#include <string>

class MNISTloader {
public:
    static int ReverseInt(int i);
    static std::vector<std::vector<float>> LoadImages(const std::string &filename);
    static std::vector<int> LoadLabels(const std::string &filename);
};