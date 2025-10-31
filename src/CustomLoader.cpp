#include "CustomLoader.h"
#include <iostream>
#include <fstream>
#include <filesystem>

std::vector<std::pair<std::vector<float>, int>> CustomLoader::LoadImages(const std::string &filePath) {
    std::vector<std::pair<std::vector<float>, int>> dataset;

    if (!std::filesystem::exists(filePath)) {
        std::cerr << "[N.N. LOAD] No save file found" << std::endl;
        return dataset;
    }

    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[N.N. LOAD] Could not open file" << std::endl;
        return dataset;
    }

    while (true) {
        int count;
        if (!in.read(reinterpret_cast<char*>(&count), sizeof(int))) break;

        std::vector<float> image(count);
        in.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(count * sizeof(float)));

        int label;
        in.read(reinterpret_cast<char*>(&label), sizeof(int));

        dataset.emplace_back(image, label);
    }

    in.close();
    return dataset;
}

void CustomLoader::SaveImage(const std::vector<std::vector<float>> &imageDrawn, const int label, const std::string &filePath, const int imageSize) {
    std::ofstream out(filePath, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[N.N. SAVE] Could not open: " << filePath << std::endl;
        return;
    }

    const auto centeredImage = CenterImage(imageDrawn, imageSize);
    const auto blurred = GaussianBlur(centeredImage);

    std::vector<float> imageDrawnVec;
    imageDrawnVec.reserve(imageDrawn.size() * imageDrawn.size());
    for (int y = 0; y < imageDrawn.size(); y++) {
        for (int x = 0; x < imageDrawn.size(); x++) {
            imageDrawnVec.push_back(blurred[y][x]);
        }
    }

    const int pixelCount = static_cast<int>(imageDrawnVec.size());
    out.write(reinterpret_cast<const char*>(&label), sizeof(int));
    out.write(reinterpret_cast<const char*>(&pixelCount), sizeof(int));
    out.write(reinterpret_cast<const char*>(imageDrawnVec.data()), pixelCount * sizeof(float));
}

std::vector<std::vector<float>> CustomLoader::CenterImage(std::vector<std::vector<float>> image, const int imageSize) {
    int top = imageSize, bottom = 0, left = imageSize, right = 0;

    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            if (image[y][x] > 0.1f) {
                top = std::min(top, y);
                bottom = std::max(bottom, y);
                left = std::min(left, x);
                right = std::max(right, x);
            }
        }
    }

    if (top >= bottom || left >= right) return image;

    const int h = bottom - top + 1;
    const int w = right - left + 1;

    std::vector temp(imageSize, std::vector(imageSize, 0.0f));

    const int offY = (28 - h) / 2;
    const int offX = (28 - w) / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            temp[offY + y][offX + x] = image[top + y][left + x];
        }
    }

    image = std::move(temp);
    return image;
}

std::vector<std::vector<float>> CustomLoader::GaussianBlur(const std::vector<std::vector<float>>& image) {
    const int size = static_cast<int>(image.size());
    std::vector out (size, std::vector(size, 0.0f));

    for (int h = 1; h < size - 1; h++) {
        for (int w = 1; w < size - 1; w++) {
            float sum = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    constexpr float kernel[3][3] {
                        {1, 2, 1},
                        {2, 4, 2},
                        {1, 2, 1}
                    };
                    sum += image[h + ky][w + kx] * kernel[ky + 1][kx + 1];
                }
            }
            out[h][w] = sum / 16.0f;
        }
    }
    return out;
}