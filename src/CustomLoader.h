#pragma once
#include <string>
#include <vector>

class CustomLoader {
public:
    static std::vector<std::pair<std::vector<float>, int>> LoadImages(const std::string &filePath);
    static void SaveImage(const std::vector<std::vector<float>> &imageDrawn, int label, const std::string &filePath, int imageSize);
    static std::vector<int> LoadLabels(const std::string &filePath);
private:
    static std::vector<std::vector<float>> CenterImage(std::vector<std::vector<float>> image, int imageSize);
    static std::vector<std::vector<float>> GaussianBlur(const std::vector<std::vector<float>>& image);
};