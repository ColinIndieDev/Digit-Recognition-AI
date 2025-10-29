#pragma once

#include <glm/glm.hpp>
#include "../CPL.h"

namespace CPL {
    struct Color;
    class Shader;

    class Line {
    public:
        glm::vec2 startPos;
        glm::vec2 endPos;
        Color color;

        explicit Line(glm::vec2 startPos, glm::vec2 endPos, const Color& color);
        ~Line();
        void Draw(const Shader& shader) const;
    private:
        unsigned int VBO{}, VAO{};
    };
}