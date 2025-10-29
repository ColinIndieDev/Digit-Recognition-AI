#include "Texture2D.h"
#include "../Shader.h"
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace CPL {
    Texture2D::Texture2D(const std::string& filePath, const glm::vec2 position, const glm::vec2 size, const Color& color, const TextureFiltering& textureFiltering) : position(position), size(size), textureSize(size), color(color) {
        const float vertices[] = {
            // positions                                                    // texture coords
            static_cast<float>(size.x), 0.0f, 0.0f,                         1.0f, 1.0f, // top right
            static_cast<float>(size.x), static_cast<float>(size.y), 0.0f,   1.0f, 0.0f, // bottom right
            0.0f, static_cast<float>(size.y), 0.0f,                         0.0f, 0.0f, // bottom left
            0.0f, 0.0f, 0.0f,                                               0.0f, 1.0f  // top left
        };
        constexpr unsigned int indices[] = {
            0, 1, 3,
            1, 2, 3
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        // Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);
        // Texture coordinates
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureFiltering == LINEAR ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureFiltering == LINEAR ? GL_LINEAR : GL_NEAREST);
        stbi_set_flip_vertically_on_load(true);
        int width, height;
        unsigned char *data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
        GLenum format = 0;
        if (channels == 1)
            format = GL_RED;
        else if (channels == 3)
            format = GL_RGB;
        else if (channels == 4)
            format = GL_RGBA;
        if (data) {
            this->size.x = static_cast<float>(width);
            this->size.y = static_cast<float>(height);
            glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(format), static_cast<GLsizei>(this->size.x), static_cast<GLsizei>(this->size.y), 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else {
            Logging::Log(2, "Failed to load texture");
        }
        stbi_image_free(data);
    }
    Texture2D::Texture2D(const std::string& filePath, const glm::vec2 size, const TextureFiltering& textureFiltering) : position(0.0f), size(size), textureSize(size), color(WHITE) {
        const float vertices[] = {
            // positions                        // texture coords
            this->size.x, 0.0f, 0.0f,           1.0f, 1.0f, // top right
            this->size.x, this->size.y, 0.0f,   1.0f, 0.0f, // bottom right
            0.0f, this->size.y, 0.0f,           0.0f, 0.0f, // bottom left
            0.0f, 0.0f, 0.0f,                   0.0f, 1.0f  // top left
        };
        constexpr unsigned int indices[] = {
            0, 1, 3,
            1, 2, 3
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        // Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), static_cast<void*>(nullptr));
        glEnableVertexAttribArray(0);
        // Texture coordinates
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureFiltering == LINEAR ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureFiltering == LINEAR ? GL_LINEAR : GL_NEAREST);
        stbi_set_flip_vertically_on_load(true);
        int width, height;
        unsigned char *data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
        GLenum format = 0;
        if (channels == 1)
            format = GL_RED;
        else if (channels == 3)
            format = GL_RGB;
        else if (channels == 4)
            format = GL_RGBA;
        if (data) {
            this->size.x = static_cast<float>(width);
            this->size.y = static_cast<float>(height);
            glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(format), static_cast<GLsizei>(this->size.x), static_cast<GLsizei>(this->size.y), 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else {
            Logging::Log(2, "Failed to load texture");
        }
        stbi_image_free(data);
    }

    void Texture2D::Unload() const {
        if (texture != 0)
            glDeleteTextures(1, &texture);
        if (VAO != 0)
            glDeleteVertexArrays(1, &VAO);
        if (VBO != 0)
            glDeleteBuffers(1, &VBO);
    }

    void Texture2D::Draw(const Shader& shader) const {
        auto transform = glm::mat4(1.0f);
        const glm::vec2 center = {position.x + textureSize.x / 2, position.y + textureSize.y / 2};
        transform = glm::translate(transform, glm::vec3(center, 0.0f));
        transform = glm::rotate(transform, glm::radians(rotationAngle), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::translate(transform, glm::vec3(-center, 0.0f));

        shader.SetMatrix4fv("transform", transform);
        shader.SetVector3f("offset", glm::vec3(position, 0.0f));
        shader.SetColor("inputColor", color);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}