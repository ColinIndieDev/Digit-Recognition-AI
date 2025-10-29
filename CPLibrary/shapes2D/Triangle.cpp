#include "Triangle.h"
#include "../Shader.h"
#include "../CPL.h"

namespace CPL {
    Triangle::Triangle(const glm::vec2 pos, const glm::vec2 size, const Color color) : position(pos), size(size), color(color) {
        const float vertices[] = {
            0.0f, 0.0f, 0.0f, // left
            size.x, 0.0f, 0.0f, // right
            size.x / 2,  size.y, 0.0f // bottom
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    Triangle::~Triangle() {
        if (VAO != 0 && glIsVertexArray(VAO)) {
            glDeleteVertexArrays(1, &VAO);
            VAO = 0;
        }
        if (VBO != 0 && glIsBuffer(VBO)) {
            glDeleteBuffers(1, &VBO);
            VBO = 0;
        }
    }

    void Triangle::Draw(const Shader& shader, const bool filled) const {
        auto transform = glm::mat4(1.0f);
        const glm::vec2 center = {position.x + size.x / 2, position.y + size.y / 2};
        transform = glm::translate(transform, glm::vec3(center, 0.0f));
        transform = glm::rotate(transform, -glm::radians(rotationAngle), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::translate(transform, glm::vec3(-center, 0.0f));

        shader.SetMatrix4fv("transform", transform);
        shader.SetVector3f("offset", glm::vec3(position, 0)); // Z not required for 2D shape
        shader.SetColor("inputColor", color);
        glBindVertexArray(VAO);
        if (filled) glDrawArrays(GL_TRIANGLES, 0, 3);
        else glDrawArrays(GL_LINE_LOOP, 0, 3);
        glBindVertexArray(0);
    }
}