#include "../CPLibrary/CPLibrary.h"
#include "NeuralNetwork.h"
#include "CustomLoader.h"
#include "MNISTloader.h"
#include "TimerChrono.h"

using namespace CPL;
PRIORITIZE_GPU_BY_VENDOR

const auto trainImages = MNISTloader::LoadImages("../data/train-images.idx3-ubyte");
const auto trainLabels = MNISTloader::LoadLabels("../data/train-labels.idx1-ubyte");
const auto testImages = MNISTloader::LoadImages("../data/t10k-images.idx3-ubyte");
const auto testLabels = MNISTloader::LoadLabels("../data/t10k-labels.idx1-ubyte");
const auto customTrainImagesLabels = CustomLoader::LoadImages("custom-train-images-and-labels");
std::vector<std::vector<float>> Y;

int pixelSize = 30;
int imageSize = 28;
std::vector imageDrawn(imageSize, std::vector(imageSize, 0.0f));
bool showHeatMap = false;
std::vector<float> relevance;

void HandleInput(NeuralNetwork& network);
std::vector<std::vector<float>> CenterImage(std::vector<std::vector<float>> image);
std::vector<std::vector<float>> SmoothImage(const std::vector<std::vector<float>>& image);
std::vector<std::vector<float>> GaussianBlur(const std::vector<std::vector<float>>& image);
Color HeatColor(float v);

int main() {
    for (const int label : trainLabels) {
        std::vector oneHot(10, 0.0f);
        oneHot[label] = 1.0;
        Y.push_back(oneHot);
    }

    NeuralNetwork network(784, 64, 10);
    {
        auto timer = TimerChrono("Loading network from files took");
        network.LoadNetwork("neural_network_save");
    }
    {
        auto timer = TimerChrono("Training network took");
        network.TrainNetwork(trainImages, Y, 0.1, 100);
        /*
        std::vector<std::vector<float>> m_X;
        std::vector<std::vector<float>> m_Y;

        for (auto& [image, label] : customTrainImagesLabels) {
            m_X.push_back(image);
            std::vector y(10, 0.0f);
            y[label] = 1.0f;
            m_Y.push_back(y);
        }

        network.TrainNetwork(m_X, m_Y, 0.1f, 10);
        */
    }

    InitWindow(280 * 3, 280 * 3, "Neural Network");

    while (!WindowShouldClose()) {
        UpdateCPL();

        HandleInput(network);

        ClearBackground(showHeatMap && !relevance.empty() ? Color(150, 150, 150, 255) : BLACK);
        BeginDrawing(SHAPE_2D, false);
        for (int h = 0; h < imageSize; h++) {
            DrawLine({0, h * 30}, {GetScreenWidth(), h * 30}, Color(255, 255, 255, 50));
        }
        for (int w = 0; w < imageSize; w++) {
            DrawLine({w * 30, 0}, {w * 30, GetScreenHeight()}, Color(255, 255, 255, 50));
        }

        if (showHeatMap && !relevance.empty()) {
            for (int h = 0; h < imageSize; h++) {
                for (int w = 0; w < imageSize; w++) {
                    const float value = relevance[h * imageSize + w];
                    Color c = {
                        255,
                        255 * (1.0f - value),
                        0,
                        150
                    };
                    DrawRectangle({w * pixelSize, h * pixelSize}, {pixelSize, pixelSize}, c);
                }
            }
        }
        for (int h = 0 ; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                if (const float pixel = imageDrawn[h][w];
                    pixel != 0.0f) {
                    if (showHeatMap) DrawRectangleOutline({w * pixelSize, h * pixelSize}, {pixelSize, pixelSize}, Color{255, 255, 255, 150});
                    else DrawRectangle({w * pixelSize, h * pixelSize}, {pixelSize, pixelSize}, Color{255, 255, 255, 255});
                    }
            }
        }

        BeginDrawing(TEXT, false);
        ShowDetails();

        EndDrawing();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    CloseWindow();
}

void HandleInput(NeuralNetwork& network) {
    if (IsKeyPressedOnce(KEY_ENTER)) {
        network.TrainNetwork(trainImages, Y, 0.1, 1);
    }
    if (IsKeyPressedOnce(KEY_R)) {
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                imageDrawn[h][w] = 0.0f;
            }
        }
        relevance.clear();
    }
    if (IsMouseDown(MOUSE_BUTTON_LEFT)) {
        glm::vec2 mousePos = GetMousePosition();
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                Rectangle pixel({w * pixelSize, h * pixelSize}, {pixelSize, pixelSize}, WHITE);
                if (Circle pixelCircle(mousePos, 30.f, WHITE);
                    CheckCollisionCircleRect(pixelCircle, pixel) && imageDrawn[h][w] < 1.0f) {
                    imageDrawn[h][w] = 1.0f;
                }
            }
        }
    }
    if (IsKeyPressedOnce(KEY_SPACE)) {
        auto centeredImage = CenterImage(imageDrawn);
        auto blurred = GaussianBlur(centeredImage);
        std::vector<float> input;
        std::vector<float> smoothedInput;
        input.reserve(imageSize * imageSize);
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                input.push_back(blurred[h][w]);
            }
        }
        auto outputs = network.FeedForward(input);

        int predicted = static_cast<int>(std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())));
        relevance = network.RelevanceMap(input, predicted);
        float maxVal = *std::max_element(relevance.begin(), relevance.end());
        float minVal = *std::min_element(relevance.begin(), relevance.end());
        for (auto& v : relevance) {
            v = (v - minVal) / (maxVal - minVal);
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << "Result" << std::endl;
        std::cout << "------------------------------------------------------\n";

        for (int number = 0; number < 10; number++) {
            std::cout << number << ": " << outputs[number] * 100 << "%" << std::endl;
        }
        std::vector<float>::iterator result;
        result = std::max_element(outputs.begin(), outputs.end());
        int index = static_cast<int>(std::distance(outputs.begin(), result));

        std::cout << "------------------------------------------------------\n";
        std::cout << "The number shown is: " << index << " | Probability: " << outputs[index] * 100 << "%" << std::endl;
        std::cout << "------------------------------------------------------\n";
    }
    if (IsKeyDown(GLFW_KEY_LEFT_SHIFT)) {
        showHeatMap = true;
    }
    else {
        showHeatMap = false;
    }
    if (IsKeyPressedOnce(KEY_T)) {
        auto timer = TimerChrono("Testing network took");

        int correct = 0;
        const int total = static_cast<int>(testImages.size());

        for (int i = 0; i < total; i++) {
            auto output = network.FeedForward(testImages[i]);

            const int predicted = static_cast<int>(std::distance(output.begin(),
                                                                 std::max_element(output.begin(), output.end())));

            if (const int actual = testLabels[i];
                predicted == actual)
                correct++;
        }

        const double accuracy = 100.0 * correct / total;
        std::cout << "[N.N. TEST] Test accuracy: " << accuracy << "% (" << correct << "/" << total << ")" << std::endl;
    }
    if (IsKeyPressedOnce(KEY_W)) {
        std::cout << "[N.N. DYNAMIC TRAINER] Solution is wrong?" << std::endl;
        std::cout << "[N.N. DYNAMIC TRAINER] Enter the number between 0 and 9: " << std::endl;
        int label = 0;
        std::cin >> label;
        if (label < 0 || label > 9) {
            std::cerr << "[N.N. DYNAMIC TRAINER] Only numbers between 0 and 9 are allowed!" << std::endl;
            return;
        }

        std::cout << "[N.N. DYNAMIC TRAINER] Set the learn rate (recommended 0.01 or 0.1): " << std::endl;
        float rate = 0.0f;
        std::cin >> rate;
        if (rate < 0.0f) {
            std::cerr << "[N.N. DYNAMIC TRAINER] Only positive values are allowed!" << std::endl;
            return;
        }

        std::cout << "[N.N. DYNAMIC TRAINER] Set epochs for training: " << std::endl;
        int epochs = 0;
        std::cin >> epochs;
        if (epochs < 0) {
            std::cerr << "[N.N. DYNAMIC TRAINER] Only positive values are allowed!" << std::endl;
            return;
        }

        CustomLoader::SaveImage(imageDrawn, label, "custom-train-images-and-labels", imageSize);

        std::vector y(10, 0.0f);
        y[label] = 1.0f;
        std::vector<std::vector<float>> m_Y = {y};
        auto centeredImage = CenterImage(imageDrawn);
        auto blurred = GaussianBlur(centeredImage);
        std::vector<float> input;
        input.reserve(imageSize * imageSize);
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                input.push_back(blurred [h][w]);
            }
        }
        std::vector<std::vector<float>> m_X = {input};

        network.TrainNetwork(m_X, m_Y, rate, epochs);
    }
    if (IsKeyPressedOnce(KEY_I)) {
        std::cout << "[N.N. DYNAMIC TRAINER] Set the learn rate (recommended 0.01 or 0.1): " << std::endl;
        float rate = 0.0f;
        std::cin >> rate;
        if (rate < 0.0f) {
            std::cerr << "[N.N. TRAINER] Only positive values are allowed!" << std::endl;
            return;
        }

        std::cout << "[N.N. DYNAMIC TRAINER] Set epochs for training: " << std::endl;
        int epochs = 0;
        std::cin >> epochs;
        if (epochs < 0) {
            std::cerr << "[N.N. TRAINER] Only positive values are allowed!" << std::endl;
            return;
        }

        auto timer = TimerChrono("Training network took");
        network.TrainNetwork(trainImages, Y, rate, epochs);
    }
    if (IsKeyPressedOnce(KEY_ESCAPE)) glfwSetWindowShouldClose(window, true);
}

std::vector<std::vector<float>> CenterImage(std::vector<std::vector<float>> image) {
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

std::vector<std::vector<float>> GaussianBlur(const std::vector<std::vector<float>>& image) {
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

Color HeatColor(float v) {
    v = std::clamp(v, 0.0f, 1.0f);

    const float r = std::min(std::max(4 * (v - 0.75f), 0.0f), 1.0f);
    const float g = std::min(std::max(4 * (v - 0.25f), 0.0f), 1.0f);
    const float b = std::min(std::max(4 * (0.25f - v), 0.0f), 1.0f);

    return Color(static_cast<uint8_t>(r * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(b * 255), 150);
}

std::vector<std::vector<float>> SmoothImage(const std::vector<std::vector<float>>& image) {
    auto out = image;
    for (int h = 1; h < imageSize - 1; h++) {
        for (int w = 1; w < imageSize - 1; w++) {
            out[h][w] = (image[h][w]
                + image[h -1][w]
                + image[h + 1][w]
                + image[h][w - 1]
                + image[h][w + 1]) / 0.5f;
        }
    }
    return out;
}