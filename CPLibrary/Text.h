#pragma once
#include <map>
#include <string>
#include <glm/glm.hpp>
#include "glad/glad.h"
#include "CPL.h"

namespace CPL {
    struct Character {
        unsigned int TextureID;
        glm::ivec2 Size;
        glm::ivec2 Bearing;
        unsigned int Advance;
    };

    class Text {
    public:
        static std::map<std::string, std::map<GLchar, Character>> Fonts;

        static void Init(const std::string& fontPath, const std::string& fontName, const TextureFiltering& textureFiltering);
        static void Use(const std::string& fontName);
        static void DrawText(const Shader& shader, const std::string& text, glm::vec2 pos, float scale, const Color& color);
        static glm::vec2 GetTextSize(const std::string& fontName, const std::string& text, float scale);
    private:
        static unsigned int VAO, VBO;
        static std::string currentFont;
    };
}