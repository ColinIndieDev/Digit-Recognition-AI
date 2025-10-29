#include "Text.h"

#include <filesystem>
#include "CPL.h"
#include "Shader.h"
#include "Logging.h"
#include <ft2build.h>
#include FT_FREETYPE_H

namespace CPL {
    std::string Text::currentFont;
    std::map<std::string, std::map<GLchar, Character>> Text::Fonts;
    unsigned int Text::VAO;
    unsigned int Text::VBO;

    void Text::Init(const std::string& fontPath, const std::string& fontName, const TextureFiltering& textureFiltering) {
        // ----- Freetype ----- //
        FT_Library ft;
        if (FT_Init_FreeType(&ft)) {
            Logging::Log(2, "Could not init FreeType Library");
            exit(-1);
        }

        if (const std::string font_name = std::filesystem::path(fontPath).string();
            font_name.empty()) {
            Logging::Log(2, "Failed to load " + font_name);
            exit(-1);
        }

        FT_Face face;
        if (FT_New_Face(ft, fontPath.c_str(), 0, &face)) {
            Logging::Log(2, "Failed to load font");
            exit(-1);
        }
        FT_Set_Pixel_Sizes(face, 0, 48);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        std::map<GLchar, Character> characters;
        for (unsigned char c = 0; c < 128; c++) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
                Logging::Log(2, "Failed to load Glyph");
                continue;
            }

            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                static_cast<GLsizei>(face->glyph->bitmap.width),
                static_cast<GLsizei>(face->glyph->bitmap.rows),
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face->glyph->bitmap.buffer
            );

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureFiltering == LINEAR ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureFiltering == LINEAR ? GL_LINEAR : GL_NEAREST);

            Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                static_cast<unsigned int>(face->glyph->advance.x)
            };
            characters.insert(std::pair<char, Character>(c, character));
        }
        Fonts.insert(std::pair(fontName, characters));
        currentFont = fontName;
        glBindTexture(GL_TEXTURE_2D, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

        // ----- VAO & VBO ----- //
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        FT_Done_Face(face);
        FT_Done_FreeType(ft);
    }

    void Text::Use(const std::string& fontName) {
        if (Fonts.contains(fontName)) currentFont = fontName;
        else Logging::Log(1, "Cannot find font");
    }

    void Text::DrawText(const Shader& shader, const std::string& text, glm::vec2 pos, const float scale, const Color& color) {
        const glm::mat4 textProjection = glm::ortho(
         0.0f, static_cast<float>(SCREEN_WIDTH),
         0.0f, static_cast<float>(SCREEN_HEIGHT)
        );
        shader.SetMatrix4fv("projection", textProjection);
        shader.SetVector3f("textColor", {color.r, color.g, color.b});
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(VAO);

        for (char c : text) {
            const auto [TextureID, Size, Bearing, Advance] = Fonts[currentFont].at(c);

            const float xPos = pos.x + static_cast<float>(Bearing.x) * scale;
            const float yPos = pos.y - static_cast<float>((Size.y - Bearing.y)) * scale;
            const float width = static_cast<float>(Size.x) * scale;
            const float height = static_cast<float>(Size.y) * scale;

            const float vertices[6][4] = {
                { xPos, yPos + height, 0.0f, 0.0f},
                { xPos, yPos, 0.0f, 1.0f},
                { xPos + width, yPos, 1.0f, 1.0f},

                { xPos, yPos + height, 0.0f, 0.0f},
                { xPos + width, yPos, 1.0f, 1.0f},
                { xPos + width, yPos + height, 1.0f, 0.0f}
            };

            glBindTexture(GL_TEXTURE_2D, TextureID);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            pos.x += static_cast<float>(Advance >> 6) * scale;
        }
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glm::vec2 Text::GetTextSize(const std::string& fontName, const std::string& text, const float scale) {
        if (!Fonts.contains(fontName)) {
            Logging::Log(1, "Cannot find font");
            return glm::vec2(0.0f);
        }

        float width = 0.0f;
        float height = 0.0f;
        float maxAboveBaseline = 0.0f;
        float maxBelowBaseline = 0.0f;

        for (char c : text) {
            const Character& ch = Fonts[fontName].at(c);
            const float h = static_cast<float>(ch.Size.y) * scale;
            maxAboveBaseline = std::max(maxAboveBaseline, static_cast<float>(ch.Bearing.y) * scale);
            maxBelowBaseline = std::max(maxBelowBaseline, (h - static_cast<float>(ch.Bearing.y) * scale));
            width += static_cast<float>(ch.Advance >> 6) * scale;
        }
        height = maxAboveBaseline + maxBelowBaseline;
        return {width, height};
    }
}