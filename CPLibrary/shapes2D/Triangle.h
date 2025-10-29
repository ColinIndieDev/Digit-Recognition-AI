#pragma once

#include <glm/glm.hpp>
#include "../CPL.h"

namespace CPL {
    struct Color;
    class Shader;

    class Triangle {
    public:
        glm::vec2 position;
        glm::vec2 size;
        Color color;
        mutable float rotationAngle = 0.0f;

        explicit Triangle(glm::vec2 pos, glm::vec2 size, Color color);
        ~Triangle();
        void Draw(const Shader& shader, bool filled) const;
    private:
        unsigned int VBO{}, VAO{};
    };
}