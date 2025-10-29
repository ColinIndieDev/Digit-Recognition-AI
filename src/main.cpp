#include "../CPLibrary/CPLibrary.h"
#include "NeuralNetwork.h"
#include "MNISTloader.h"
#include "TimerChrono.h"

using namespace CPL;
PRIORITIZE_GPU_BY_VENDOR

const auto trainImages = MNISTloader::LoadImages("../data/train-images.idx3-ubyte");
const auto trainLabels = MNISTloader::LoadLabels("../data/train-labels.idx1-ubyte");
const auto testImages = MNISTloader::LoadImages("../data/t10k-images.idx3-ubyte");
const auto testLabels = MNISTloader::LoadLabels("../data/t10k-labels.idx1-ubyte");
std::vector<std::vector<float>> Y;

int pixelSize = 30;
int imageSize = 28;
std::vector imageDrawn(imageSize, std::vector(imageSize, 0.0f));

void HandleInput(NeuralNetwork& network);
std::vector<std::vector<float>> CenterImage(std::vector<std::vector<float>> image);

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
        //auto timer = TimerChrono("Training network took");
        //network.TrainNetwork(trainImages, Y, 0.01, 100);
    }

    InitWindow(280 * 3, 280 * 3, "Neural Network");

    while (!WindowShouldClose()) {
        UpdateCPL();

        HandleInput(network);

        ClearBackground(BLACK);
        BeginDrawing(SHAPE_2D, false);
        for (int h = 0 ; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                if (const float pixel = imageDrawn[h][w];
                    pixel != 0.0f) DrawRectangle({w * pixelSize, h * pixelSize}, {pixelSize, pixelSize}, Color{pixel * 255, pixel * 255, pixel * 255, 255});
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
    }
    if (IsMouseDown(MOUSE_BUTTON_LEFT)) {
        glm::vec2 mousePos = GetMousePosition();
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                Rectangle pixel({w * pixelSize, h * pixelSize}, {pixelSize, pixelSize}, WHITE);
                //if (CheckCollisionVec2Rect(mousePos, pixel) && imageDrawn[h][w] < 1.0f) {
                //    imageDrawn[h][w] = 1.0f;
                //}
                if (Circle pixelCircle(mousePos, 20.f, WHITE);
                    CheckCollisionCircleRect(pixelCircle, pixel) && imageDrawn[h][w] < 1.0f) {
                    imageDrawn[h][w] = 1.0f;
                }
            }
        }
    }
    if (IsKeyPressedOnce(KEY_SPACE)) {
        std::vector<float> imageDrawnVec;
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                imageDrawnVec.push_back(CenterImage(imageDrawn)[h][w]);
            }
        }
        std::vector<float> outputs = network.FeedForward(imageDrawnVec);

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
    if (IsKeyPressedOnce(KEY_I)) {
        auto timer = TimerChrono("Training network took");
        network.TrainNetwork(trainImages, Y, 0.1, 1);
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