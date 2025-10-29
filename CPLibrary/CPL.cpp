#include "CPL.h"
#include "Audio.h"
#include "shapes2D/Triangle.h"
#include "shapes2D/Rectangle.h"
#include "shapes2D/Circle.h"
#include "shapes2D/Line.h"
#include "Shader.h"
#include "Text.h"
#include "shapes2D/Texture2D.h"
#include "timers/TimerManager.h"
#include "stb_image.h"
#include "Path.h"

namespace CPL {
    Shader shapeShader;
    Shader textShader;
    Shader textureShader;

    void UpdateCPL() {
        UpdateInput();
        CalculateDeltaTime();
        CalculateFPS();
        TimerManager::Update(GetDeltaTime());
        AudioManager::Update();
    }

    void ShowDetails() {
        const GLubyte* renderer = glGetString(GL_RENDERER);
        const GLubyte* vendor = glGetString(GL_VENDOR);
        const GLubyte* version = glGetString(GL_VERSION);
        const std::string rendererString(reinterpret_cast<const char*>(renderer));
        const std::string vendorString(reinterpret_cast<const char*>(vendor));
        const std::string versionString(reinterpret_cast<const char*>(version));

        BeginDrawing(TEXT, false);
        const std::string fpsText = "FPS: " + std::to_string(GetFPS());
        DrawTextShadow({0, 25}, {2, 2}, 0.3, fpsText, WHITE, DARK_GRAY);
        const std::string vendorText = "Vendor: " + vendorString;
        DrawTextShadow({0, 60}, {2, 2}, 0.3, vendorText, WHITE, DARK_GRAY);
        const std::string rendererText = "GPU: " + rendererString;
        DrawTextShadow({0, 95}, {2, 2}, 0.3, rendererText, WHITE, DARK_GRAY);
        const std::string versionText = "Version: " + versionString;
        DrawTextShadow({0, 130}, {2, 2}, 0.3, versionText, WHITE, DARK_GRAY);
        EndDrawing();
    }

    bool CheckCollisionRects(const Rectangle& one, const Rectangle& two) {
        const bool collisionX = one.position.x + one.size.x >= two.position.x &&
            two.position.x + two.size.x >= one.position.x;
        const bool collisionY = one.position.y + one.size.y >= two.position.y &&
            two.position.y + two.size.y >= one.position.y;

        return collisionX && collisionY;
    }
    bool CheckCollisionCircleRect(const Circle& one, const Rectangle& two) {
        const glm::vec2 circleCenter = one.position;
        const glm::vec2 rectCenter = two.position + two.size * 0.5f;
        const glm::vec2 halfExtents = two.size * 0.5f;
        const glm::vec2 difference = circleCenter - rectCenter;
        const glm::vec2 clamped = glm::clamp(difference, -halfExtents, halfExtents);
        const glm::vec2 closest = rectCenter + clamped;
        const glm::vec2 delta = closest - circleCenter;

        return glm::length(delta) <= one.radius;
    }
    bool CheckCollisionVec2Rect(const glm::vec2& one, const Rectangle& two) {
        return two.position.x < one.x && one.x < two.position.x + two.size.x
            && two.position.y < one.y && one.y < two.position.y + two.size.y;
    }

    void InitWindow(const int width, const int height, const char* title) {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        SCREEN_WIDTH = width;
        SCREEN_HEIGHT = height;
        projection = glm::ortho(
            0.0f, static_cast<float>(width),
            static_cast<float>(height), 0.0f,
            -1.0f, 1.0f
        );

        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (window == nullptr) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(-1);
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            exit(-1);
        }

        InitShaders();
        Text::Init(Path::GetAssetPath("assets/fonts/default.ttf"), "defaultFont", NEAREST);
        AudioManager::Init();

        // ----- For the font & 2D textures ----- //
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    }

    void SetWindowIcon(const std::string& filePath) {
        int width, height, channels;
        stbi_load(filePath.c_str(), &width, &height, &channels, 0);
        int requiredComponents = 0;
        if (channels == 3) requiredComponents = 3;
        else if (channels == 4) requiredComponents = 4;
        GLFWimage images[1];
        images[0].pixels = stbi_load(filePath.c_str(), &images[0].width, &images[0].height, nullptr, requiredComponents);
        if (images[0].pixels) {
            glfwSetWindowIcon(window, 1, images);
            stbi_image_free(images[0].pixels);
        }
        else {
            Logging::Log(2, "Failed to load icon");
        }
    }

    void CloseWindow() {
        glfwTerminate();
        AudioManager::Close();
    }

    void InitShaders() {
        shapeShader = Shader(Path::GetAssetPath("CPLibrary/shaders/shader.vert").c_str(), Path::GetAssetPath("CPLibrary/shaders/shader.frag").c_str());
        textShader = Shader(Path::GetAssetPath("CPLibrary/shaders/text.vert").c_str(), Path::GetAssetPath("CPLibrary/shaders/text.frag").c_str());
        textureShader = Shader(Path::GetAssetPath("CPLibrary/shaders/texture.vert").c_str(), Path::GetAssetPath("CPLibrary/shaders/texture.frag").c_str());
    }

    void BeginDrawing(const DrawModes& mode, const bool mode2D) {
        Shader shader{};
        if (mode == SHAPE_2D) shader = shapeShader;
        else if (mode == TEXT) {
            shader = textShader;
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            Text::Use("defaultFont");
        }
        else if (mode == TEXTURE_2D) shader = textureShader;

        shader.Use();
        const glm::mat4 view = camera.GetViewMatrix();
        const glm::mat4 viewProjection = projection * view;
        shader.SetMatrix4fv("projection", mode2D ? viewProjection : projection);
    }

    void DrawTriangle(const glm::vec2 position, const glm::vec2 size, const Color& color) {
        const auto triangle = Triangle(position, size, color);
        triangle.Draw(shapeShader, true);
    }
    void DrawTriangleRotated(const glm::vec2 position, const glm::vec2 size, const float angle, const Color& color) {
        const auto triangle = Triangle(position, size, color);
        triangle.rotationAngle = angle;
        triangle.Draw(shapeShader, true);
    }
    void DrawTriangleOutline(const glm::vec2 position, const glm::vec2 size, const Color& color) {
        const auto triangle = Triangle(position, size, color);
        triangle.Draw(shapeShader, false);
    }
    void DrawTriangleRotOut(const glm::vec2 position, const glm::vec2 size, const float angle, const Color& color) {
        const auto triangle = Triangle(position, size, color);
        triangle.rotationAngle = angle;
        triangle.Draw(shapeShader, false);
    }

    void DrawRectangle(const glm::vec2 position, const glm::vec2 size, const Color& color) {
        const auto rectangle = Rectangle(position, size, color);
        rectangle.Draw(shapeShader, true);
    }
    void DrawRectangleRotated(const glm::vec2 position, const glm::vec2 size, const float angle, const Color& color) {
        const auto rectangle = Rectangle(position, size, color);
        rectangle.rotationAngle = angle;
        rectangle.Draw(shapeShader, true);
    }
    void DrawRectangleOutline(const glm::vec2 position, const glm::vec2 size, const Color& color) {
        const auto rectangle = Rectangle(position, size, color);
        rectangle.Draw(shapeShader, false);
    }
    void DrawRectangleRotOut(const glm::vec2 position, const glm::vec2 size, const float angle, const Color& color) {
        const auto rectangle = Rectangle(position, size, color);
        rectangle.rotationAngle = angle;
        rectangle.Draw(shapeShader, false);
    }

    void DrawCircle(const glm::vec2 position, const float radius, const Color& color) {
        const auto circle = Circle(position, radius, color);
        circle.Draw(shapeShader);
    }
    void DrawCircleOutline(const glm::vec2 position, const float radius, const Color& color) {
        const auto circle = Circle(position, radius, color);
        circle.DrawOutline(shapeShader);
    }

    void DrawLine(const glm::vec2 startPos, const glm::vec2 endPos, const Color& color) {
        const auto line = Line(startPos, endPos, color);
        line.Draw(shapeShader);
    }

    void DrawTexture2D(Texture2D* texture, const glm::vec2 position, const Color& color) {
        texture->position = position;
        texture->color = color;
        texture->Draw(textureShader);
    }
    void DrawTexture2DRotated(Texture2D* texture, const glm::vec2 position, const float angle, const Color& color) {
        texture->position = position;
        texture->color = color;
        texture->rotationAngle = angle;
        texture->Draw(textureShader);
    }
    void DrawTex2DCpy(Texture2D texture, const glm::vec2 position, const Color& color) {
        texture.position = position;
        texture.color = color;
        texture.Draw(textureShader);
    }

    void DrawText(const glm::vec2 position, const float scale, const std::string& text, const Color& color) {
        Text::DrawText(textShader, text, position, scale, color);
    }
    void DrawTextShadow(const glm::vec2 position, const glm::vec2 shadowOffset, const float scale, const std::string& text, const Color& color, const Color& shadowColor) {
        Text::DrawText(textShader, text, {position.x + shadowOffset.x, position.y - shadowOffset.y}, scale, shadowColor);
        Text::DrawText(textShader, text, position, scale, color);
    }
}
