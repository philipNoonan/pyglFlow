#version 430
    in vec3 newColor;
    in vec2 outTexCoords;
    out vec4 outColor;
	
    layout (binding = 0) uniform sampler2D samplerTex;
    layout (binding = 1) uniform sampler2D samplerProcessedTex;

	uniform int sliderR;
	uniform int sliderG;
	uniform int sliderB;

    uniform int renderType;

    void main()
    {

        vec3 col = vec3(0,0,0);

        if (renderType == 0) {
            // newcolor just shows you how to pass values through the shader stages
            //outColor = vec4(newColor, 1.0f);
            col = texture(samplerTex, outTexCoords, float(sliderR)).xyz  * 1.0f;

        }
        else if (renderType == 1) {
            col.xy = texture(samplerProcessedTex, outTexCoords, float(sliderR)).xy * 1.0f;

        }

		outColor = vec4(col, 1.0f);
    }