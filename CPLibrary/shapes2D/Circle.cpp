#include "Circle.h"
#include "../CPL.h"
#include <cmath>
#include <vector>

#include "../Shader.h"

namespace CPL {
    Circle::Circle(const glm::vec2 pos, const float radius, const Color color) : position(pos), radius(radius), color(color) {
        std::vector<float> vertices;
        const int segments = std::ceil(radius);
        vertices.reserve(segments * 3);
        for (int i = 0; i <= segments; i++) {
            const float theta = 2 * static_cast<float>(M_PI)  / static_cast<float>(segments) * static_cast<float>(i);
            float x = 0.0f + radius * std::cos(theta);
            float y = 0.0f +radius * std::sin(theta);
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(0);
        }

        // ----- Outline circle ----- //
        glGenVertexArrays(1, &outlineVAO);
        glGenBuffers(1, &outlineVBO);
        glBindVertexArray(outlineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, outlineVBO);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        // ----- Filled circle ----- //
        for (int i = 0; i < 3; i++) { vertices.push_back(0);}
        vertexCount = static_cast<int>(vertices.size()) / 3;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    Circle::~Circle() {
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
    }

    void Circle::Draw(const Shader& shader) const {
        auto transform = glm::mat4(1.0f);
        const glm::vec2 center = {position.x, position.y};
        transform = glm::translate(transform, glm::vec3(center, 0.0f));
        transform = glm::rotate(transform, -glm::radians(rotationAngle), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::translate(transform, glm::vec3(-center, 0.0f));

        shader.SetMatrix4fv("transform", transform);
        shader.SetVector3f("offset", glm::vec3(position, 0.0f));
        shader.SetColor("inputColor", color);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_FAN, 0, vertexCount);
        glBindVertexArray(0);
    }

    void Circle::DrawOutline(const Shader& shader) const {
        auto transform = glm::mat4(1.0f);
        const glm::vec2 center = {position.x, position.y};
        transform = glm::translate(transform, glm::vec3(center, 0.0f));
        transform = glm::rotate(transform, -glm::radians(rotationAngle), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::translate(transform, glm::vec3(-center, 0.0f));

        shader.SetMatrix4fv("transform", transform);
        shader.SetVector3f("offset", glm::vec3(position, 0.0f));
        shader.SetColor("inputColor", color);

        glBindVertexArray(outlineVAO);
        glDrawArrays(GL_LINE_LOOP, 0, vertexCount - 1);
        glBindVertexArray(0);
    }
}