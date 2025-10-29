#include "Line.h"

#include "../Shader.h"

namespace CPL {
    Line::Line(const glm::vec2 startPos, const glm::vec2 endPos, const Color& color) : startPos(startPos), endPos(endPos), color(color) {
        const float vertices[] = {
            startPos.x, startPos.y, 0.0f,
            endPos.x, endPos.y, 0.0f,
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
    Line::~Line() {
        if (VAO != 0 && glIsVertexArray(VAO)) {
            glDeleteVertexArrays(1, &VAO);
            VAO = 0;
        }
        if (VBO != 0 && glIsBuffer(VBO)) {
            glDeleteBuffers(1, &VBO);
            VBO = 0;
        }
    }


    void Line::Draw(const Shader& shader) const {
        shader.SetMatrix4fv("transform", glm::mat4(1.0f));
        shader.SetVector3f("offset", glm::vec3(0.0f));
        shader.SetColor("inputColor", color);
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

}