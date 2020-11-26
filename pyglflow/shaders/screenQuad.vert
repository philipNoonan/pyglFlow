#version 430
    in layout(location = 0) vec3 position;
    in layout(location = 1) vec3 color;
    in layout(location = 2) vec2 inTexCoords;
    out vec3 newColor;
    out vec2 outTexCoords;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        newColor = color;
        outTexCoords = vec2(inTexCoords.x, 1.0f - inTexCoords.y);
    }