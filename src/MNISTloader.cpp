#include "MNISTloader.h"

#include <fstream>
#include <vector>
#include <stdexcept>

int MNISTloader::ReverseInt(const int i) {
    const unsigned char c1 = i & 255;
    const unsigned char c2 = i >> 8 & 255;
    const unsigned char c3 = i >> 16 & 255;
    const unsigned char c4 = i >> 24 & 255;
    return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + c4;
}

std::vector<std::vector<float>> MNISTloader::LoadImages(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("[MNISTloader] Cannot open file: " + filename);

    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);

    std::vector images(number_of_images, std::vector<float>(n_rows * n_cols));
    for (int i = 0; i < number_of_images; ++i) {
        for (int j = 0; j < n_rows * n_cols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
    return images;
}

std::vector<int> MNISTloader::LoadLabels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("[MNISTloader] Cannot open file: " + filename);

    int magic_number = 0, number_of_items = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
    magic_number = ReverseInt(magic_number);
    number_of_items = ReverseInt(number_of_items);

    std::vector<int> labels(number_of_items);
    for (int i = 0; i < number_of_items; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }
    return labels;
}