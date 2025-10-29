#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <random>
#include "Colors.h"
#include "KeyInputs.h"
#include "Logging.h"

#define PRIORITIZE_GPU_BY_VENDOR extern "C" { __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001; __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1; }

namespace CPL {
    enum DrawModes {
        SHAPE_2D,
        TEXTURE_2D,
        TEXT,
    };
    enum TextureFiltering {
        NEAREST,
        LINEAR,
    };

    struct Color {
        float r, g, b, a;
    };

    struct Timer;
    class TimerManager;

    class Shader;
    class Triangle;
    class Rectangle;
    class Circle;
    class Line;
    class Texture2D;

    struct Character;
    class Text;

    inline unsigned int SCREEN_WIDTH;
    inline unsigned int SCREEN_HEIGHT;
    inline glm::mat4 projection;

    extern Shader shapeShader;
    extern Shader textShader;
    extern Shader textureShader;

    inline std::mt19937 gen{std::random_device{}()};

    inline std::unordered_map<int, bool> keyStates;
    inline std::unordered_map<int, bool> prevKeyStates;
    inline std::unordered_map<int, bool> mouseButtons;
    inline std::unordered_map<int, bool> prevMouseButtons;

    inline GLFWwindow* window;

    struct Audio;
    class AudioManager;

    struct Camera2D {
        glm::vec2 position{0.0f};
        float zoom = 1.0f;
        float rotation = 0.0f;

        [[nodiscard]] glm::mat4 GetViewMatrix() const {
            auto view = glm::mat4(1.0f);
            view = glm::translate(view, glm::vec3(-glm::vec2{position.x - static_cast<float>(SCREEN_WIDTH) / 2.0f, position.y - static_cast<float>(SCREEN_HEIGHT) / 2.0f}, 0.0f));
            view  = glm::scale(view, glm::vec3(zoom, zoom, 1.0f));
            view = glm::rotate(view, glm::radians(rotation), glm::vec3(0, 0, 1));
            return view;
        }
    };

    inline Camera2D camera;

    void UpdateCPL();

    bool CheckCollisionRects(const Rectangle& one, const Rectangle& two);
    bool CheckCollisionCircleRect(const Circle& one, const Rectangle& two);
    bool CheckCollisionVec2Rect(const glm::vec2& one, const Rectangle& two);

    void InitShaders();

    void DrawTriangle(glm::vec2 position, glm::vec2 size, const Color& color);
    void DrawTriangleRotated(glm::vec2 position, glm::vec2 size, float angle, const Color& color);
    void DrawTriangleOutline(glm::vec2 position, glm::vec2 size, const Color& color);
    void DrawTriangleRotOut(glm::vec2 position, glm::vec2 size, float angle, const Color& color);
    void DrawRectangle(glm::vec2 position, glm::vec2 size, const Color& color);
    void DrawRectangleRotated(glm::vec2 position, glm::vec2 size, float angle, const Color& color);
    void DrawRectangleOutline(glm::vec2 position, glm::vec2 size, const Color& color);
    void DrawRectangleRotOut(glm::vec2 position, glm::vec2 size, float angle, const Color& color);
    void DrawCircle(glm::vec2 position, float radius, const Color& color);
    void DrawCircleOutline(glm::vec2 position, float radius, const Color& color);
    void DrawLine(glm::vec2 startPos, glm::vec2 endPos, const Color& color);
    void DrawTexture2D(Texture2D* texture, glm::vec2 position, const Color& color);
    void DrawTexture2DRotated(Texture2D* texture, glm::vec2 position, float angle, const Color& color);
    void DrawTex2DCpy(Texture2D texture, glm::vec2 position, const Color& color);

    void DrawText(glm::vec2 position, float scale, const std::string& text, const Color& color);
    void DrawTextShadow(glm::vec2 position, glm::vec2 shadowOffset, float scale, const std::string& text, const Color& color, const Color& shadowColor);

    inline void ClearBackground(const Color& color) {
        glClearColor(color.r / 255, color.g / 255, color.b / 255, color.a / 255);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void BeginDrawing(const DrawModes& mode, bool mode2D);

    inline void EndDrawing() {
        glUseProgram(0);
    }

    // [[maybe_unused]] so CLion doesn't annoy me with redundant window
    inline void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, const int width, const int height) {
        glViewport(0, 0, width, height);
    }

    inline double lastTime = 0.0;
    inline int nbFrames = 0;
    inline int FPS;
    inline void CalculateFPS() {
        const double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) {
            FPS = nbFrames;
            nbFrames = 0;
            lastTime += 1.0;
        }
    }
    inline int GetFPS() {
        return FPS;
    }

    inline float deltaTime = 0;
    inline float lastFrame = 0;
    inline float timeScale = 1.0f;

    inline void CalculateDeltaTime() {
        const auto currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = (currentFrame - lastFrame) * timeScale;
        lastFrame = currentFrame;
    }

    inline float GetDeltaTime() { return deltaTime; }
    inline float GetTime() { return static_cast<float>(glfwGetTime()); }

    void ShowDetails();

    void InitWindow(int width, int height, const char* title);
    void SetWindowIcon(const std::string& filePath);
    void CloseWindow();

    inline int WindowShouldClose() {
        return glfwWindowShouldClose(window);
    }

    inline float GetScreenWidth() {
        return static_cast<float>(SCREEN_WIDTH);
    }
    inline float GetScreenHeight() {
        return static_cast<float>(SCREEN_HEIGHT);
    }

    inline int RandInt(const int min, const int max) {
        std::uniform_int_distribution dist(min, max);
        return dist(gen);
    }
    inline float RandFloat(const float min, const float max) {
        std::uniform_real_distribution dist(min, max);
        return dist(gen);
    }
    inline bool RandByPercentInt(const int percent)
    {
        std::uniform_int_distribution dist(1, 100);
        return dist(gen) <= percent;
    }
    inline bool RandByPercentFloat(const float percent)
    {
        std::uniform_real_distribution dist(0.0f, 100.0f);
        return dist(gen) <= percent;
    }

    inline void UpdateInput() {
        prevKeyStates = keyStates;
        for (int key = GLFW_KEY_SPACE; key <= GLFW_KEY_LAST; key++) {
            keyStates[key] = glfwGetKey(window, key) == GLFW_PRESS;
        }

        prevMouseButtons = mouseButtons;
        for (int button = GLFW_MOUSE_BUTTON_1; button <= GLFW_MOUSE_BUTTON_LAST; button++) {
            mouseButtons[button] = glfwGetMouseButton(window, button) == GLFW_PRESS;
        }
    }

    inline bool IsKeyDown(const int key) {
        return keyStates[key];
    }
    inline bool IsKeyUp(const int key) {
        return !keyStates[key];
    }
    inline bool IsKeyPressedOnce(const int key) {
        return keyStates[key] && !prevKeyStates[key];
    }
    inline bool IsKeyReleased(const int key) {
        return !keyStates[key] && prevKeyStates[key];
    }

    inline bool IsMouseDown(const int button) {
        return mouseButtons[button];
    }
    inline bool IsMousePressedOnce(const int button) {
        return mouseButtons[button] && !prevMouseButtons[button];
    }
    inline bool IsMouseReleased(const int button) {
        return !mouseButtons[button] && prevMouseButtons[button];
    }
    inline glm::vec2 GetMousePosition() {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        return {x, y};
    }
    inline glm::vec2 GetMousePositionWorld() {
        const glm::vec2 screenCenter = { GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f };
        glm::vec2 world;
        world.x = (GetMousePosition().x - screenCenter.x) / camera.zoom + camera.position.x;
        world.y = (GetMousePosition().y - screenCenter.y) / camera.zoom + camera.position.y;
        return world;
    }
}