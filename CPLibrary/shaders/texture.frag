#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform vec4 inputColor;
uniform sampler2D ourTexture;

void main()
{
    vec4 textureColor = texture(ourTexture, TexCoord);
    if (textureColor.a < 0.1) discard;
    FragColor = inputColor / 255 * textureColor;
}