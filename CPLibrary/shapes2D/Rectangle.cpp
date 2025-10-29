#include "Rectangle.h"
#include "../CPL.h"
#include "../Shader.h"

namespace CPL {
    Rectangle::Rectangle(const glm::vec2 pos, const glm::vec2 size, const Color color) : position(pos), size(size), color(color) {
        const float vertices[] = {
            size.x, 0.0f, 0.0f,  // top right
            size.x, size.y, 0.0f,  // bottom right
            0.0f, size.y, 0.0f,  // bottom left
            0.0f,  0.0f, 0.0f   // top left
        };
        constexpr unsigned int indices[] = {
            0, 1, 3,  // first Triangle
            1, 2, 3   // second Triangle
        };
        constexpr unsigned int outlineIndices[] = {
            0, 1, 2, 3
        };

        // ----- Filled Rectangle ----- //
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        // ----- Outline Rectangle ----- //
        glGenVertexArrays(1, &outlineVAO);
        glGenBuffers(1, &outlineVBO);
        glGenBuffers(1, &outlineEBO);
        glBindVertexArray(outlineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, outlineVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, outlineEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(outlineIndices), outlineIndices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    Rectangle::~Rectangle() {
        if (VAO != 0 && glIsVertexArray(VAO)) {
            glDeleteVertexArrays(1, &VAO);
            VAO = 0;
        }
        if (VBO != 0 && glIsBuffer(VBO)) {
            glDeleteBuffers(1, &VBO);
            VBO = 0;
        }
        if (outlineVAO != 0 && glIsVertexArray(outlineVAO)) {
            glDeleteVertexArrays(1, &outlineVAO);
            outlineVAO = 0;
        }
        if (outlineVBO != 0 && glIsBuffer(outlineVBO)) {
            glDeleteBuffers(1, &outlineVBO);
            outlineVBO = 0;
        }
        if (EBO != 0 && glIsBuffer(EBO)) {
            glDeleteBuffers(1, &EBO);
            EBO = 0;
        }
        if (outlineEBO != 0 && glIsBuffer(outlineEBO)) {
            glDeleteBuffers(1, &outlineEBO);
            outlineEBO = 0;
        }
    }


    void Rectangle::Draw(const Shader& shader, const bool filled) const {
        auto transform = glm::mat4(1.0f);
        const glm::vec2 center = {position.x + size.x / 2, position.y + size.y / 2};
        transform = glm::translate(transform, glm::vec3(center, 0.0f));
        transform = glm::rotate(transform, glm::radians(rotationAngle), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::translate(transform, glm::vec3(-center, 0.0f));

        shader.SetMatrix4fv("transform", transform);
        shader.SetVector3f("offset", glm::vec3(position, 0.0f));
        shader.SetColor("inputColor", color);
        if (filled) {
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        }
        else {
            glBindVertexArray(outlineVAO);
            glDrawElements(GL_LINE_LOOP, 6, GL_UNSIGNED_INT, nullptr);
        }
        glBindVertexArray(0);
    }
}